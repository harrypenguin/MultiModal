import math
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from losses.SpecLoss import forward_loss
from models.MyTimm import Block, generate_attn_mask, PatchEmbed1D
from utils.DataProcessing import generate_rest_indices
from utils.PositionalEmbedding import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from utils.Scheduler import CosineWarmupScheduler
from utils.Visualization import visualize


class MaskedAutoencoderViT(pl.LightningModule):
    """ Masked Autoencoder with VisionTransformer backbone, copied from https://github.com/facebookresearch/mae/blob/main/models_mae.py 
    """

    def __init__(
        self,
        spec_dim=7781,
        patch_size=31,
        left_patches=10,
        right_patches=10,
        embed_dim=768,
        merged_depth=6,
        merged_num_heads=6,
        s_depth=1,
        e_depth=1,
        s_num_heads=1,
        e_num_heads=1,
        decoder_embed_dim=384,
        decoder_depth=2,
        decoder_num_heads=6,
        decoder_MLP_coefficient=4,
        lr=2e-4,
        warmup_epoch=100,
        max_epochs=3000,
        batch_size=16,
        mlp_ratio=4.0,
        mask_ratio=0.75,
        patch_scheme={"patch_sizes": [1], "mask_ratios": [0.75]},
        scatter_term=1,
        log_regularizer=1.0,
        lam_grad=0.0,
        lam_curv=0.0,
        lam_fft=0.0,
        lam_topk=0.0,
        topk_frac=0.10,
        lam_spiky=0.0,
        spiky_tau=0.8,
        lam_under=0.0,
        lam_sigma_right=0.0,
        sigma_quantile=0.75,
        lam_img_sigma_masked=0.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed1d = PatchEmbed1D(spec_dim, patch_size, embed_dim)
        self.img_patch = 16
        self.num_img_channels = 6
        self.patch_embedimg = PatchEmbed(img_size=128, patch_size=self.img_patch, in_chans=1, embed_dim=embed_dim)
        self.num_patches1d = self.patch_embed1d.num_patches
        self.num_patchesimg = self.patch_embedimg.num_patches * self.num_img_channels
        self.left_patches = left_patches
        self.right_patches = right_patches

        # --- image positional embeddings (spatial + channel + modality) ---
        self.num_img_spatial = self.patch_embedimg.num_patches
        img_grid_size = int(math.sqrt(self.num_img_spatial))
        if img_grid_size * img_grid_size != self.num_img_spatial:
            raise ValueError(f"Image patch grid is not square: {self.num_img_spatial}")

        self.register_buffer(
            "img_channel_embed",
            self._build_fixed_channel_embed(self.num_img_channels, embed_dim),
            persistent=False,
        )
        self.img_modality_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.img_e_modality_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "img_spatial_pos_embed",
            get_2d_sincos_pos_embed(embed_dim, img_grid_size, img_grid_size),
            persistent=False,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 10000 + 1, embed_dim), requires_grad=False)

        self.s_attn = nn.ModuleList([
            Block(embed_dim, s_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(s_depth)
        ])
        self.e_attn = nn.ModuleList([
            Block(embed_dim, e_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(e_depth)
        ])
        self.img_attn = nn.ModuleList([
            Block(embed_dim, s_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(s_depth)
        ])
        self.img_e_attn = nn.ModuleList([
            Block(embed_dim, e_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(e_depth)
        ])

        self.merged_blocks = nn.ModuleList([
            Block(2 * embed_dim, merged_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(merged_depth)
        ]) # Attention blocks for merged spectra and images
        self.norm = norm_layer(embed_dim)
        self.merged_norm = norm_layer(2 * embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim * 2, decoder_embed_dim, bias=True)

        self.spec_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 10000 + 1, decoder_embed_dim), requires_grad=False)

        # --- decoder image positional embeddings (spatial + channel + modality) ---
        self.register_buffer(
            "decoder_img_channel_embed",
            self._build_fixed_channel_embed(self.num_img_channels, decoder_embed_dim),
            persistent=False,
        )
        self.decoder_img_modality_embed = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer(
            "decoder_img_spatial_pos_embed",
            get_2d_sincos_pos_embed(decoder_embed_dim, img_grid_size, img_grid_size),
            persistent=False,
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, patch_size),
        )

        self.decoder_e_estimator = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, patch_size),
        )

        self.decoder_pred_img = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, self.img_patch * self.img_patch),
        )

        self.decoder_pred_img_e = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, self.img_patch * self.img_patch),
        )

        refiner_hidden = 32
        self.decoder_img_refiner = nn.Sequential(
            nn.Conv2d(self.num_img_channels, refiner_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refiner_hidden, refiner_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refiner_hidden, refiner_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refiner_hidden, self.num_img_channels, kernel_size=3, padding=1),
        )
        self.decoder_img_e_refiner = nn.Sequential(
            nn.Conv2d(self.num_img_channels, refiner_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refiner_hidden, refiner_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refiner_hidden, refiner_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refiner_hidden, self.num_img_channels, kernel_size=3, padding=1),
        )
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio
        self.mask_ratio_img = 0.75
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.chunk_size = 1
        self.patch_scheme = patch_scheme
        self.warmup_epoch = warmup_epoch

        self.lambda_min_obs = 3600 - 0.8 * self.patch_size * self.left_patches
        self.lambda_max_obs = 9824 + 0.8 * self.patch_size * self.right_patches
        self.spec_dim = spec_dim

        # ---------------------------------------------------------------------------
        # Loss function parameters
        self.scatter_term = scatter_term
        self.log_regularizer = log_regularizer
        self.lam_grad = lam_grad
        self.lam_curv = lam_curv
        self.lam_fft = lam_fft
        self.lam_topk = lam_topk
        self.topk_frac = topk_frac
        self.lam_spiky = lam_spiky
        self.spiky_tau = spiky_tau
        self.lam_under = lam_under
        self.lam_sigma_right = lam_sigma_right
        self.sigma_quantile = sigma_quantile
        self.lam_img_sigma_masked = lam_img_sigma_masked

        # extras

        self.coord_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.decoder_coord_mlp = nn.Sequential(
            nn.Linear(2, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
        )

        self.img_spatial_pos_embed = get_2d_sincos_pos_embed(self.hparams.embed_dim, img_grid_size, img_grid_size)
        self.decoder_img_spatial_pos_embed = get_2d_sincos_pos_embed(
            self.hparams.decoder_embed_dim, img_grid_size, img_grid_size
        )

        self.initialize_weights()

    def initialize_weights(self):
        ###
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], 10000, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 10000, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        ### freeze sin cos PE

        # initialization
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed1d.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.spec_mask_token, std=0.02)
        torch.nn.init.normal_(self.img_mask_token, std=0.02)

        # torch.nn.init.normal_(self.img_channel_embed.weight, std=.02)
        torch.nn.init.normal_(self.img_modality_embed, std=0.02)
        torch.nn.init.normal_(self.img_e_modality_embed, std=0.02)
        # torch.nn.init.normal_(self.decoder_img_channel_embed.weight, std=.02)
        torch.nn.init.normal_(self.decoder_img_modality_embed, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _build_fixed_channel_embed(self, num_channels, embed_dim):
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even for fixed channel embeddings, got {embed_dim}")
        channel_ids = torch.arange(num_channels, dtype=torch.float32).unsqueeze(1)
        dim_half = embed_dim // 2
        omega = torch.arange(dim_half, dtype=torch.float32)
        omega = 1.0 / (10000.0 ** (omega / dim_half))
        out = channel_ids * omega.unsqueeze(0)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_image_pos_embed(self, dtype, device):
        spatial = self.img_spatial_pos_embed.to(device=device, dtype=dtype)
        spatial = spatial.expand(self.num_img_channels, -1, -1)

        channel = self.img_channel_embed.to(device=device, dtype=dtype)
        channel = channel.unsqueeze(1).expand(-1, self.num_img_spatial, -1)

        img_pos = (spatial + channel).reshape(1, self.num_img_channels * self.num_img_spatial, -1)
        return img_pos

    def get_decoder_image_pos_embed(self, dtype, device):
        spatial = self.decoder_img_spatial_pos_embed.to(device=device, dtype=dtype)
        spatial = spatial.expand(self.num_img_channels, -1, -1)

        channel = self.decoder_img_channel_embed.to(device=device, dtype=dtype)
        channel = channel.unsqueeze(1).expand(-1, self.num_img_spatial, -1)

        img_pos = (spatial + channel).reshape(1, self.num_img_channels * self.num_img_spatial, -1)
        return img_pos

    def _continuous_2d_sincos(self, xy_patch_units, embed_dim, dtype, device):
        # Used to generate PEs for spectrum based on fibre location, to match image sin cos PE
        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
        xy = xy_patch_units.to(device=device, dtype=dtype)
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        dim_each = embed_dim // 4
        omega = torch.arange(dim_each, device=device, dtype=dtype)
        omega = 1.0 / (10000.0 ** (omega / dim_each))
        out_x = x * omega.unsqueeze(0)
        out_y = y * omega.unsqueeze(0)
        return torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=-1)

    def _build_relative_patch_coords(self, xy_pix, dtype, device):
        # Returns relative coords of image patch to centre patch in patch units
        gh, gw = self.patch_embedimg.grid_size
        p_h, p_w = self.patch_embedimg.patch_size
        H, W = self.patch_embedimg.img_size

        row_centers = torch.arange(gh, device=device, dtype=dtype) * p_h + (p_h - 1) / 2.0
        col_centers = torch.arange(gw, device=device, dtype=dtype) * p_w + (p_w - 1) / 2.0

        x_centers = col_centers - (W - 1) / 2.0
        y_centers = (H - 1) / 2.0 - row_centers

        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")
        patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

        xy = xy_pix.to(device=device, dtype=dtype)
        rel_xy = patch_xy.unsqueeze(0) - xy.unsqueeze(1)
        return rel_xy / float(p_w)

    def _img_tokens_to_image(self, tokens):
        B = tokens.shape[0]
        gh, gw = self.patch_embedimg.grid_size
        p_h, p_w = self.patch_embedimg.patch_size
        x = tokens.view(B, self.num_img_channels, gh * gw, p_h * p_w)
        x = x.view(B, self.num_img_channels, gh, gw, p_h, p_w)
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(B, self.num_img_channels, gh * p_h, gw * p_w)
        return x

    def _image_to_img_tokens(self, img):
        B = img.shape[0]
        gh, gw = self.patch_embedimg.grid_size
        p_h, p_w = self.patch_embedimg.patch_size
        x = img.view(B, self.num_img_channels, gh, p_h, gw, p_w)
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(B, self.num_img_channels, gh * gw, p_h * p_w)
        return x.reshape(B, self.num_patchesimg, p_h * p_w)

    def _refine_img_tokens(self, tokens, refiner):
        img = self._img_tokens_to_image(tokens)
        img = refiner(img)
        return self._image_to_img_tokens(img)

    def forward_encoder(self, s, e, img, img_e, z, xy_pix, mask_ratio, chunk_size):
        s = self.patch_embed1d(s.unsqueeze(-1))
        e = self.patch_embed1d(e.unsqueeze(-1))

        # I'm going to hardcode 128 size assumption here for now
        xy_grid_x = (xy_pix[:, 0] + 64.0) / self.img_patch
        xy_grid_y = (64.0 - xy_pix[:, 1]) / self.img_patch
        xy_grid = torch.stack([xy_grid_x, xy_grid_y], dim=-1)

        xy_pe = self._continuous_2d_sincos(xy_grid, self.hparams.embed_dim, s.dtype, s.device).unsqueeze(1)

        s = s + xy_pe
        e = e + xy_pe

        rel_coords = self._build_relative_patch_coords(xy_pix, dtype=img.dtype, device=img.device)
        coord_emb = self.coord_mlp(rel_coords)
        img = torch.cat([self.patch_embedimg(img[:, i:i + 1]) + coord_emb for i in range(img.size(1))], dim=1)
        img_e = torch.cat([self.patch_embedimg(img_e[:, i:i + 1]) + coord_emb for i in range(img_e.size(1))], dim=1)

        img_pos_embed = self.get_image_pos_embed(dtype=img.dtype, device=img.device)
        img = img + img_pos_embed + self.img_modality_embed.to(dtype=img.dtype)
        img_e = img_e + img_pos_embed + self.img_e_modality_embed.to(dtype=img_e.dtype)

        deredshifted_start_indices, deredshifted_end_indices = generate_rest_indices(s, z, patch_size=self.patch_size)
        pos_table = self.pos_embed[:, 1:, :].squeeze(0)
        pe_start = pos_table[deredshifted_start_indices]
        pe_end = pos_table[deredshifted_end_indices]
        s = s + pe_start + pe_end
        e = e + pe_start + pe_end

        attn_mask, token_mask = generate_attn_mask(self.chunk_size, self.mask_ratio, self.num_patches1d + 1, device=s.device)
        attn_mask_img, token_mask_img = generate_attn_mask(1, self.mask_ratio_img, self.num_patchesimg, device=s.device)

        cls_token = self.cls_token + self.pos_embed[:, 0, :]
        cls_tokens = cls_token.expand(s.shape[0], -1, -1)
        s = torch.cat((cls_tokens, s), dim=1)
        e = torch.cat((cls_tokens, e), dim=1)

        for blk in self.s_attn:
            s = blk(s, attn_mask, token_mask)
        s = self.norm(s)

        for blk in self.e_attn:
            e = blk(e, attn_mask, token_mask)
        e = self.norm(e)

        for blk in self.img_attn:
            img = blk(img, attn_mask_img, token_mask_img)
        img = self.norm(img)

        for blk in self.img_e_attn:
            img_e = blk(img_e, attn_mask_img, token_mask_img)
        img_e = self.norm(img_e)

        x = torch.cat([s, e], dim=-1)
        x_img = torch.cat([img, img_e], dim=-1)
        x = torch.cat([x, x_img], dim=1)

        overall_attn_mask = torch.block_diag(attn_mask, attn_mask_img)
        overall_token_mask = torch.cat([token_mask, token_mask_img], dim=0)

        for blk in self.merged_blocks:
            x = blk(x, overall_attn_mask, overall_token_mask)
        x = self.merged_norm(x)

        return x, overall_attn_mask, overall_token_mask, deredshifted_start_indices, deredshifted_end_indices

    def forward_decoder(self, x, token_mask, z, xy_pix):
        x = self.decoder_embed(x)

        token_mask = token_mask.unsqueeze(0).expand(x.shape[0], -1)
        spec_mask = token_mask[:, :-self.num_patchesimg]
        img_mask = token_mask[:, -self.num_patchesimg:]

        x_spec = x[:, :-self.num_patchesimg, :]
        x_img = x[:, -self.num_patchesimg:, :]
        x_spec[spec_mask] = self.spec_mask_token.to(x.dtype)
        x_img[img_mask] = self.img_mask_token.to(x.dtype)
        x = torch.cat([x_spec, x_img], dim=1)

        B, _, _ = x.shape
        left_tokens = self.spec_mask_token.repeat(B, self.left_patches, 1)
        right_tokens = self.spec_mask_token.repeat(B, self.right_patches, 1)

        cls_token = x[:, :1, :]
        main_seq = x[:, 1:-self.num_patchesimg, :]
        img_seq = x[:, -self.num_patchesimg:, :]
        padded_seq = torch.cat([left_tokens, main_seq, right_tokens, img_seq], dim=1)
        x = torch.cat([cls_token, padded_seq], dim=1)

        pos_table = self.decoder_pos_embed[:, 1:, :].squeeze(0)

        spec_len = x.shape[1] - 1 - self.num_patchesimg
        x_spec = x[:, 1:1 + spec_len, :]
        x_img = x[:, -self.num_patchesimg:, :]

        x_for_pe = x[:, :1 + spec_len, :]
        deredshifted_start_indices, deredshifted_end_indices = generate_rest_indices(
            x_for_pe,
            z,
            lambda_min_obs=self.lambda_min_obs,
            patch_size=self.patch_size,
        )

        pe_start = pos_table[deredshifted_start_indices]
        pe_end = pos_table[deredshifted_end_indices]
        x_spec = x_spec + pe_start[:, 1:, :] + pe_end[:, 1:, :]
        xy_grid_x = (xy_pix[:, 0] + 64.0) / self.img_patch
        xy_grid_y = (64.0 - xy_pix[:, 1]) / self.img_patch
        xy_grid = torch.stack([xy_grid_x, xy_grid_y], dim=-1)
        xy_pe = self._continuous_2d_sincos(xy_grid, self.hparams.decoder_embed_dim, x.dtype, x.device).unsqueeze(1)
        x_spec = x_spec + xy_pe

        img_pos = self.get_decoder_image_pos_embed(dtype=x.dtype, device=x.device)
        rel_coords = self._build_relative_patch_coords(xy_pix, dtype=x.dtype, device=x.device)
        coord_emb = self.decoder_coord_mlp(rel_coords).repeat(1, self.num_img_channels, 1)
        x_img = x_img + img_pos + self.decoder_img_modality_embed.to(dtype=x.dtype) + coord_emb

        x = torch.cat([x[:, :1, :], x_spec, x_img], dim=1)

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        s = self.decoder_pred(x[:, :-self.num_patchesimg, :])
        e = self.decoder_e_estimator(x[:, :-self.num_patchesimg, :])
        img = self.decoder_pred_img(x[:, -self.num_patchesimg:, :])
        img_e = self.decoder_pred_img_e(x[:, -self.num_patchesimg:, :])
        img = self._refine_img_tokens(img, self.decoder_img_refiner)
        img_e = self._refine_img_tokens(img_e, self.decoder_img_e_refiner)

        s = s[:, 1:, :].view(s.shape[0], -1)
        e = e[:, 1:, :].view(e.shape[0], -1)
        img = img.view(img.shape[0], -1)
        img_e = img_e.view(img_e.shape[0], -1)

        return s, e, img, img_e

    def forward(self, spec, weig, error, img, weig_img, error_img, z, xy_pix):
        latent, _, token_mask, _, _ = self.forward_encoder(
            spec, error, img, error_img, z, xy_pix, self.mask_ratio, self.chunk_size
        )
        pred, error, pred_img, error_img = self.forward_decoder(latent, token_mask, z, xy_pix)

        offset = self.left_patches * self.patch_size
        spec_loss, img_loss, total_loss = forward_loss(
            pred[:, offset:offset + self.spec_dim],
            spec,
            weig,
            error[:, offset:offset + self.spec_dim],
            pred_img,
            img,
            weig_img,
            error_img,
            token_mask[:-self.num_patchesimg].long(),
            img_mask=token_mask[-self.num_patchesimg:].long(),
            num_patches1d=self.num_patches1d,
            left_patches=self.left_patches,
            patch_size=self.patch_size,
            img_patch=self.img_patch,
            num_img_channels=self.num_img_channels,
            num_img_patches=self.patch_embedimg.num_patches,
            weight=self.scatter_term,
            regularizer=self.log_regularizer,
            lam_grad=self.lam_grad,
            lam_curv=self.lam_curv,
            lam_topk=self.lam_topk,
            lam_fft=self.lam_fft,
            topk_frac=self.topk_frac,
            lam_spiky=self.lam_spiky,
            spiky_tau=self.spiky_tau,
            lam_under=self.lam_under,
            lam_sigma_right=self.lam_sigma_right,
            lam_img_sigma_masked=self.lam_img_sigma_masked,
        )
        return spec_loss, img_loss, total_loss, pred, error, pred_img, error_img, token_mask

    def training_step(self, batch, batch_idx):
        zero_loss = sum((p.sum() * 0.0) for p in self.parameters() if p.requires_grad)

        if batch is None:
            print(f"Empty batch at step {batch_idx}; using zero loss")
            return zero_loss

        self.chunk_size, self.mask_ratio = self.sample_patching()
        _, mask_ratio_img = self.sample_patching()
        if mask_ratio_img == self.mask_ratio and mask_ratio_img == 1:
            mask_ratio_img = 0.9
        self.mask_ratio_img = mask_ratio_img
        self.log("chunk_size", self.chunk_size)
        self.log("mask_ratio", self.mask_ratio)
        self.log("mask_ratio_img", self.mask_ratio_img)

        x, spec, weig, error, img, img_w, img_e, z, xy_pix = batch
        spec_loss, img_loss, total_loss, _, _, _, _, _ = self.forward(spec, weig, error, img, img_w, img_e, z, xy_pix)

        if not torch.isfinite(total_loss):
            print(f"Non-finite loss at step {batch_idx}; using zero loss")
            return zero_loss

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("spec_loss", spec_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("img_loss", img_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("grad_norm", self._grad_norm(), on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        self.chunk_size, self.mask_ratio = self.sample_patching()
        _, mask_ratio_img = self.sample_patching()
        if mask_ratio_img == self.mask_ratio and mask_ratio_img == 1:
            mask_ratio_img = 0.9
        self.mask_ratio_img = mask_ratio_img

        x, spec, weig, error, img, img_w, img_e, z, xy_pix = batch
        spec_loss, img_loss, total_loss, spec_pred, error_pred, pred_img, error_img, token_mask = self.forward(
            spec, weig, error, img, img_w, img_e, z, xy_pix
        )

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_spec_loss", spec_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_img_loss", img_loss, on_step=True, on_epoch=True, prog_bar=False)

        if batch_idx == 0:
            token_mask = token_mask.unsqueeze(0).expand(spec.size(0), -1)
            mask_spec = token_mask[:, :-self.num_patchesimg].long()
            mask_img = token_mask[:, -self.num_patchesimg:].long()
            visualize(self, spec, error, spec_pred, error_pred, img, img_e, pred_img, error_img, mask_spec, mask_img)

        return total_loss

    def sample_patching(self):
        """
        Sample (patch_size, mask_ratio) such that each patch_size is always
        matched with its corresponding mask_ratio.

        Expected `self.patch_scheme` layout::

            self.patch_scheme = {
                "patch_sizes": [64, 128, 256],      # required
                "mask_ratios": [0.3, 0.5, 0.7],     # required
                "probs":       [0.2, 0.6, 0.2]      # optional weights, len = N
            }

        * If 'probs' is omitted, pairs are sampled uniformly.
        * All three lists (patch_sizes, mask_ratios, probs) must share length N.
        """
        sch = self.patch_scheme
        try:
            patch_sizes = sch["patch_sizes"]
            mask_ratios = sch["mask_ratios"]
        except KeyError as e:
            raise KeyError(f"Missing required key: {e.args[0]}")

        if len(patch_sizes) != len(mask_ratios):
            raise ValueError("patch_sizes and mask_ratios must have the same length")

        probs = sch.get("probs")
        if probs and len(probs) != len(patch_sizes):
            raise ValueError("probs length must match patch_sizes length")

        idx = random.choices(range(len(patch_sizes)), weights=probs, k=1)[0]
        return patch_sizes[idx], mask_ratios[idx]

    def _grad_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Run cosine schedule per optimizer step instead of per epoch.
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", self.max_epochs))
        total_steps = max(1, total_steps)
        warmup_steps = int(self.warmup_epoch * total_steps / max(1, int(self.max_epochs)))
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

        self.lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=warmup_steps,
            max_iters=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
