# A more self-contained training file from before modularizing the code

import os
import sys
import math
import random

sys.path.append("..")

import numpy as np
import pandas
import matplotlib.pyplot as plt
import zarr
import wandb
from pytorch_lightning.loggers import WandbLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning.pytorch import seed_everything

from timm.models.vision_transformer import PatchEmbed

from utils.DataProcessing import (
    generate_rest_indices,
    get_extreme_mask,
    get_kernel,
    safe_collate,
    smooth_data,
    MultimodalDataset,
    CreateMultimodalDataLoadersIter,
)
from utils.AstroImageFunctions import unwise_to_rgb, flux_to_rgb, make_rgb
from utils.PositionalEmbedding import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from models.MyTimm import Block, generate_attn_mask, PatchEmbed1D


class MaskedAutoencoderViT(pl.LightningModule):
    """ Masked Autoencoder with VisionTransformer backbone, copied from https://github.com/facebookresearch/mae/blob/main/models_mae.py 
    """
    def __init__(self, spec_dim=7781, patch_size=31, left_patches=10, right_patches=10,
                 embed_dim=768, merged_depth=6, merged_num_heads=6,
                 s_depth=1, e_depth=1, s_num_heads = 1, e_num_heads = 1,
                 decoder_embed_dim=384, decoder_depth=2, decoder_num_heads=6, 
                 decoder_MLP_coefficient=4, lr=2e-4,
                 warmup_epoch=100, max_epochs=3000, batch_size=16,
                 mlp_ratio=4., mask_ratio=0.75, patch_scheme={"patch_sizes":[1], "mask_ratios": [0.75]},
                 # extra loss function parameters
                 scatter_term=1, log_regularizer=1.0, lam_grad=0.0, lam_curv=0.0, lam_fft=0.0, lam_topk=0.0, topk_frac=0.10,
                 lam_spiky=0.0, spiky_tau=0.8, lam_under=0.0, lam_sigma_right=0.0, sigma_quantile=0.75,
                 lam_img_sigma_masked=0.0,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False):
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

        # self.img_channel_embed = nn.Embedding(self.num_img_channels, embed_dim)
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
        self.pos_embed = nn.Parameter(torch.zeros(1, 10000 + 1, embed_dim), requires_grad = False)  # z aware PE

        self.s_attn = nn.ModuleList([
            Block(embed_dim, s_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(s_depth)])
        
        self.e_attn = nn.ModuleList([
            Block(embed_dim, e_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(e_depth)])

        self.img_attn = nn.ModuleList([
            Block(embed_dim, s_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(s_depth)])

        self.img_e_attn = nn.ModuleList([
            Block(embed_dim, e_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(e_depth)])

        self.merged_blocks = nn.ModuleList([
            Block(2 * embed_dim, merged_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(merged_depth)]) # Attention blocks for merged spectra and images
        self.norm = norm_layer(embed_dim)
        self.merged_norm = norm_layer(2 * embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim * 2, decoder_embed_dim, bias=True)

        self.spec_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 10000 + 1, decoder_embed_dim), requires_grad = False)  # z aware PE

        # --- decoder image positional embeddings (spatial + channel + modality) ---
        # self.decoder_img_channel_embed = nn.Embedding(self.num_img_channels, decoder_embed_dim)
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
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, patch_size)
        )

        self.decoder_e_estimator = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, patch_size)
        )

        self.decoder_pred_img = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, self.img_patch * self.img_patch)
        )

        self.decoder_pred_img_e = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, decoder_MLP_coefficient * decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_MLP_coefficient * decoder_embed_dim, self.img_patch * self.img_patch)
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

        # hardcoded wavelengths based on DESI instrument specs
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
        self.decoder_img_spatial_pos_embed = get_2d_sincos_pos_embed(self.hparams.decoder_embed_dim, img_grid_size, img_grid_size)


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
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.spec_mask_token, std=.02)
        torch.nn.init.normal_(self.img_mask_token, std=.02)

        # torch.nn.init.normal_(self.img_channel_embed.weight, std=.02)
        torch.nn.init.normal_(self.img_modality_embed, std=.02)
        torch.nn.init.normal_(self.img_e_modality_embed, std=.02)
        # torch.nn.init.normal_(self.decoder_img_channel_embed.weight, std=.02)
        torch.nn.init.normal_(self.decoder_img_modality_embed, std=.02)

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
        B = xy_pix.shape[0]

        gh, gw = self.patch_embedimg.grid_size
        p_h, p_w = self.patch_embedimg.patch_size
        H, W = self.patch_embedimg.img_size

        row_centers = torch.arange(gh, device=device, dtype=dtype) * p_h + (p_h - 1) / 2.0
        col_centers = torch.arange(gw, device=device, dtype=dtype) * p_w + (p_w - 1) / 2.0

        x_centers = col_centers - (W - 1) / 2.0
        y_centers = (H - 1) / 2.0 - row_centers

        yy, xx = torch.meshgrid(y_centers, x_centers, indexing='ij')
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

        xy_pe = self._continuous_2d_sincos(
            xy_grid, self.hparams.embed_dim, s.dtype, s.device
        ).unsqueeze(1)

        s = s + xy_pe
        e = e + xy_pe

        rel_coords = self._build_relative_patch_coords(xy_pix, dtype=img.dtype, device=img.device)
        coord_emb = self.coord_mlp(rel_coords)
        img = torch.cat([self.patch_embedimg(img[:, i:i+1]) + coord_emb for i in range(img.size(1))], dim=1)
        img_e = torch.cat([self.patch_embedimg(img_e[:, i:i+1]) + coord_emb for i in range(img_e.size(1))], dim=1)

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
        xy_pe = self._continuous_2d_sincos(
            xy_grid, self.hparams.decoder_embed_dim, x.dtype, x.device
        ).unsqueeze(1)
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

    # utils for extra spec loss terms
    def _grad1(self, x):
        # x: (B, L)
        return x[..., 1:] - x[..., :-1]

    def _grad2(self, x):
        return x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]

    def _normalized_weighted_mse(self, x_hat, x, w, eps=1e-6):
        """
        Compute normalized weighted MSE loss.
        x_hat: (B, L) - predicted spectrum
        x: (B, L) - ground truth spectrum
        w: (B, L) - weights (1 / sigma^2)
        eps: float - small constant for numerical stability
        """
        num = (w * (x_hat - x).pow(2)).sum(dim=-1)
        den = w.sum(dim=-1).clamp_min(eps)
        return (num / den).mean()

    def _spiky_weight(self, x, tau=0.8, eps=1e-6):
        """
        Compute a weight that emphasizes high gradients.
        tau: float - quantile threshold for spike detection
        """
        g = self._grad1(x).abs()
        g = F.pad(g, (1, 0))  # align to length L
        g = g / (g.mean(dim=-1, keepdim=True) + eps)
        thresh = torch.quantile(g, tau, dim=-1, keepdim=True)
        return (g > thresh).float() * g.clamp_min(0)

    def _topk_mse(self, x_hat, x, kfrac=0.1):
        """
        Compute the top-k MSE loss.
        kfrac: float - fraction of top-k pixels to consider
        """
        err = (x_hat - x).pow(2)
        L = err.shape[-1]
        k = max(1, int(L * float(kfrac)))
        topk, _ = torch.topk(err, k, dim=-1, largest=True, sorted=False)
        return topk.mean()

    def _fft_hf_loss(self, x_hat, x, lam=1.0, eps=1e-6):
        """
        Compute the high-frequency loss using FFT.
        """
        x_hat32 = x_hat.to(torch.float32)
        x32     = x.to(torch.float32)
        Xh = torch.fft.rfft(x_hat32, dim=-1)
        X  = torch.fft.rfft(x32, dim=-1)
        K  = X.shape[-1]
        k  = torch.arange(K, device=x.device, dtype=torch.float32)
        w  = (k / k.max().clamp_min(1)).clamp_min(eps).view(1, K)
        return lam * (w * (Xh - X).abs()).mean()

    def _asym_under_penalty(self, x_hat, x, lam=1.0):
        """
        Compute an asymmetric penalty for under-predictions.
        This loss penalizes cases where x_hat is less than x.
        """
        under = (x - x_hat).clamp_min(0.0)
        return lam * under.pow(2).mean()

    def forward_loss(
        self, x_hat, x, w, log_s, img_hat, img, weig_img, error_img, mask, img_mask=None, *,
        # original knobs
        weight=1.0,
        regularizer=1.0,
        eps=1e-6,
        # extras (masked-only)
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
    ):
        """
        mask: [B, P or P+1], 0=keep, 1=remove. If P+1, the first is CLS and is dropped.
        Extra losses are computed ONLY on masked pixels within the *centre window*
        used to form x_hat/x (i.e., [offset : offset+spec_dim]).
        """
        B, L = x.shape

        # ---- Align patch mask to the centre window used for x_hat/x ----
        P_in = mask.shape[-1]
        P = getattr(self, "num_patches1d", None)
        if P is None:
            # Fallback: infer P from patch_size and a known full length
            P = P_in - 1 if P_in * getattr(self, "patch_size", 1) > L else P_in

        # Drop CLS if present
        if P_in == P + 1:
            patch_mask = mask[..., 1:]
        else:
            patch_mask = mask

        psize = int(getattr(self, "patch_size", max(1, L // P)))
        # Build full pixel mask over all patches, then crop to center window
        pixel_mask_full = patch_mask.repeat_interleave(psize, dim=-1)  # (B, P*psize)
        # Compute center slice bounds to match x_hat/x
        offset = int(getattr(self, "left_patches", 0)) * psize
        end = offset + L
        # Ensure length and clamp bounds defensively
        if pixel_mask_full.shape[-1] < end:
            # pad with zeros (visible) if needed
            pad = end - pixel_mask_full.shape[-1]
            pixel_mask_full = torch.nn.functional.pad(pixel_mask_full, (0, pad))
        pixel_mask = pixel_mask_full[..., offset:end].to(x.dtype)  # (B, L), 1=masked

        # ---- Base loss over ALL pixels ----
        w_safe = torch.nan_to_num(torch.clamp(w, min=eps), nan=eps, posinf=1.0 / eps, neginf=eps)
        log_s = torch.nan_to_num(log_s, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        sigma_hat_sq = weight * torch.exp(log_s).clamp(min=eps).pow(2)
        denom = (1.0 / w_safe + sigma_hat_sq).clamp(min=eps)

        if torch.any(torch.isnan(denom)):
            print("NaNs detected in denominator")
        if torch.any(denom <= 0):
            print(f"Non-positive denominator. Min: {denom.min().item()}")

        sq_error = (x_hat - x).pow(2)
        base_pixel = 0.5 * (sq_error / denom + regularizer * torch.log(denom.clamp_min(eps)))
        loss = base_pixel.mean()

        # # Early exit if there are no masked pixels in the centre window
        # if pixel_mask.sum() == 0:
        #     return loss

        # ---- Helper: masked mean aligned to finite-diff sizes ----
        def mask_mean(t, m, shrink=None, eps=1e-6):
            if shrink == 'grad1':
                # both neighbors must be masked
                m_use = (m[..., 1:] * m[..., :-1])
            elif shrink == 'grad2':
                # three-point stencil all masked
                m_use = (m[..., 2:] * m[..., 1:-1] * m[..., :-2])
            else:
                m_use = m
            num = (t * m_use).sum(dim=-1)
            den = m_use.sum(dim=-1).clamp_min(eps)
            return (num / den).mean()

        # ---- Extras only on masked pixels ---- from ChatGPT
        if lam_grad > 0.0:
            d1 = (self._grad1(x_hat) - self._grad1(x)).abs()          # (B, L-1)
            Lg = mask_mean(d1, pixel_mask, shrink='grad1')             # uses (B, L-1) mask
            loss = loss + lam_grad * Lg

        if lam_curv > 0.0:
            d2 = (self._grad2(x_hat) - self._grad2(x)).abs()          # (B, L-2)
            Lc = mask_mean(d2, pixel_mask, shrink='grad2')             # uses (B, L-2) mask
            loss = loss + lam_curv * Lc

        if lam_fft > 0.0:
            # Masked residual HF: confine emphasis to masked region
            r = (x_hat - x) * pixel_mask
            loss = loss + self._fft_hf_loss(r, torch.zeros_like(r), lam=lam_fft)

        if lam_topk > 0.0:
            err = (x_hat - x).pow(2)
            masked_err = torch.where(pixel_mask > 0, err, torch.full_like(err, -float('inf')))
            k = max(1, int(L * float(topk_frac)))
            topk, _ = torch.topk(masked_err, k, dim=-1, largest=True, sorted=False)
            Ltopk = torch.clamp(topk, min=0).mean()
            loss = loss + lam_topk * Ltopk

        if lam_spiky > 0.0:
            W = self._spiky_weight(x, tau=spiky_tau, eps=eps) * pixel_mask
            loss = loss + lam_spiky * self._normalized_weighted_mse(x_hat, x, W, eps=eps)

        if lam_under > 0.0:
            under = (x - x_hat).clamp_min(0.0).pow(2)
            Lunder = mask_mean(under, pixel_mask)
            loss = loss + lam_under * Lunder

        if lam_sigma_right > 0.0:
            sigma_hat = torch.exp(log_s)
            thr = torch.quantile(sigma_hat.detach(), sigma_quantile)
            over = (sigma_hat - thr).clamp_min(0.0).pow(2)
            Lsig = mask_mean(over, pixel_mask)
            loss = loss + lam_sigma_right * Lsig

        B = img.size(0)
        P = self.img_patch
        C = self.num_img_channels
        N = self.patch_embedimg.num_patches
        # (B, 98304) -> (B, C, N, P*P) -> (B, N, C, P*P) -> (B, N, C*P*P)
        img_hat   = img_hat.view(B, C, N, P*P).permute(0, 2, 1, 3).reshape(B, N, C*P*P)
        error_img = error_img.view(B, C, N, P*P).permute(0, 2, 1, 3).reshape(B, N, C*P*P)
        
        weig_img = weig_img.unfold(2,P,P).unfold(3,P,P).permute(0,2,3,1,4,5).reshape(img.size(0), N, C*P*P)
        img   = img.unfold(2,P,P).unfold(3,P,P).permute(0,2,3,1,4,5).reshape(img.size(0), N, C*P*P)

        weig_img = torch.nan_to_num(torch.clamp(weig_img, min=eps), nan=eps, posinf=1.0 / eps, neginf=eps)
        error_img = torch.nan_to_num(error_img, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        img_loss = 0.5 * (((img_hat - img)**2) / (1.0 / weig_img + weight * torch.exp(error_img).clamp_min(eps)**2)
                  + regularizer * torch.log((1.0 / weig_img + weight * torch.exp(error_img).clamp_min(eps)**2))).mean()

        if lam_img_sigma_masked > 0.0 and img_mask is not None:
            if img_mask.dim() == 1:
                img_mask = img_mask.unsqueeze(0).expand(B, -1)

            img_mask = img_mask.to(device=error_img.device, dtype=error_img.dtype).unsqueeze(-1)

            error_img_tok = error_img.view(B, N, C, P * P).permute(0, 2, 1, 3).reshape(B, C * N, P * P)
            sigma_img = torch.exp(error_img_tok).clamp_min(eps)

            denom = (img_mask.sum() * (P * P)).clamp_min(1.0)
            img_sigma_penalty = (sigma_img.pow(2) * img_mask).sum() / denom
            img_loss = img_loss + lam_img_sigma_masked * img_sigma_penalty

        # img_loss = F.mse_loss(img_hat, img)
        # Turning off img error head temporarily

        return loss, img_loss, loss + img_loss


    def forward(self, spec, weig, error, img, weig_img, error_img, z, xy_pix):
        latent, attn_mask, token_mask, deredshifted_start_indices, deredshifted_end_indices = self.forward_encoder(
            spec, error, img, error_img, z, xy_pix, self.mask_ratio, self.chunk_size
        )
        pred, error, pred_img, error_img = self.forward_decoder(latent, token_mask, z, xy_pix)

        offset = self.left_patches * self.patch_size
        spec_loss, img_loss, total_loss = self.forward_loss(
            pred[:, offset : offset + self.spec_dim],
            spec,
            weig,
            error[:, offset : offset + self.spec_dim],
            pred_img,
            img,
            weig_img,
            error_img,
            token_mask[:-self.num_patchesimg].long(),
            img_mask=token_mask[-self.num_patchesimg:].long(),
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

        x, spec, weig, error, img, img_w, img_e, z, xy_pix = batch
        spec_loss, img_loss, total_loss, spec_pred, error_pred, pred_img, error_img, token_mask = self.forward(spec, weig, error, img, img_w, img_e, z, xy_pix)

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_spec_loss", spec_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_img_loss", img_loss, on_step=True, on_epoch=True, prog_bar=False)

        if batch_idx == 0:
            token_mask = token_mask.unsqueeze(0).expand(spec.size(0), -1)
            mask_spec = token_mask[:, :-self.num_patchesimg].long()
            mask_img = token_mask[:, -self.num_patchesimg:].long()
            self.visualize(spec, error, spec_pred, error_pred, img, img_e, pred_img, error_img, mask_spec, mask_img)

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

        # Pick a single index, respecting optional weights
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
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': self.lr_scheduler,
            'interval': 'step',
            'frequency': 1,
        }}
    
    def visualize(self, spec: torch.Tensor, error: torch.Tensor, spec_pred: torch.Tensor,
                  error_pred: torch.Tensor, img: torch.Tensor, img_error_true: torch.Tensor,
                  pred_img: torch.Tensor, error_img: torch.Tensor, mask_spec: torch.Tensor, mask_img: torch.Tensor,
                  i: int = 0, nsmooth: int = 3):
        kernel = get_kernel(nsmooth)

        # Extract relevant sample
        target = spec[i].detach().cpu().numpy()
        target_error = error[i].detach().cpu().numpy()
        pred_mean = spec_pred[i].detach().cpu().numpy()
        pred_sigma = np.exp(error_pred[i].detach().cpu().numpy())
        patch_mask_spec = mask_spec[i].detach().cpu()
        patch_mask_img = mask_img[i].detach().cpu()

        # Smooth signal and error
        target_smooth = smooth_data(target, kernel)
        target_error_smooth = smooth_data(target_error, kernel)

        # Compute intervals
        pred_upper = pred_mean + pred_sigma
        pred_lower = pred_mean - pred_sigma
        data_upper = target_smooth + target_error_smooth
        data_lower = target_smooth - target_error_smooth

        # Plot masked reconstruction
        fig, ax = plt.subplots(figsize=(10, 3))
        offset = self.left_patches * self.patch_size
        x_range = np.arange(offset, offset + len(target_smooth))
        ax.plot(x_range, target_smooth, label="Target (smoothed)", color="orange", linewidth=1)
        ax.fill_between(x_range, data_lower, data_upper, color="orange", alpha=0.3, label="±σ data (smoothed)")
        ax.plot(pred_mean, label="Prediction", color="blue")
        ax.fill_between(range(len(pred_mean)), pred_lower, pred_upper, color="blue", alpha=0.3, label="±σ predicted")
        patch_len = self.patch_size
        for j, masked in enumerate(patch_mask_spec):
            if masked:
                start = offset + j * patch_len
                end = offset + (j + 1) * patch_len
                ax.axvspan(start, end, color='red', alpha=0.05)
        ax.set_title(f"Patch Size = {self.patch_size*self.chunk_size}, {self.mask_ratio} (Val Sample {i})")
        ax.set_xlabel("Pixel index")
        ax.set_ylabel("Value")
        ax.set_ylim(-3, 15)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        @rank_zero_only
        def log_reconstruction():
            wandb.log({"reconstruction": wandb.Image(fig)})

        log_reconstruction()
        

        N = self.patch_embedimg.num_patches
        gh, gw = self.patch_embedimg.grid_size
        H, W = img.shape[-2], img.shape[-1]

        p = pred_img[i].view(self.num_img_channels, N, self.img_patch, self.img_patch)
        p = p.view(self.num_img_channels, gh, gw, self.img_patch, self.img_patch)
        pred_img_formatted = p.permute(0, 1, 3, 2, 4).reshape(self.num_img_channels, H, W).detach().cpu().numpy()

        pe = error_img[i].view(self.num_img_channels, N, self.img_patch, self.img_patch)
        pe = pe.view(self.num_img_channels, gh, gw, self.img_patch, self.img_patch)
        pred_error_formatted = np.exp(pe.permute(0, 1, 3, 2, 4).reshape(self.num_img_channels, H, W).detach().cpu().numpy())
        
        img_formatted = img[i].detach().cpu().numpy()
        img_error_formatted = img_error_true[i].detach().cpu().numpy()
        pred_rgb = make_rgb(pred_img_formatted[[0, 1, 3]], "ls_grz") 
        img_rgb = make_rgb(img_formatted[[0, 1, 3]], "ls_grz") 
        
        # mask: (384,) -> (6,64) -> (6,8,8)
        # m = patch_mask_img.float().view(6, 64).view(6, 8, 8)

        # # opacity per spatial patch = fraction of channels masked
        # alpha_patch = m.mean(dim=0).cpu().numpy()                 # (8,8)
        # alpha128 = np.kron(alpha_patch, np.ones((16, 16)))        # (128,128)

        fig2, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(img_rgb)
        axs[0].set_title("True")
        axs[0].axis("off")

        axs[1].imshow(pred_rgb)
        # axs[1].imshow(alpha128, cmap="Reds", alpha=alpha128)
        axs[1].set_title("Masked Reconstruction")
        axs[1].axis("off")

        plt.tight_layout()

        @rank_zero_only
        def log_images():
            wandb.log({"image_recon": wandb.Image(fig2)})

        log_images()

        mask_grid = patch_mask_img.float().view(self.num_img_channels, gh, gw).cpu().numpy()
        rows = [img_formatted, pred_img_formatted, img_error_formatted, pred_error_formatted]
        row_labels = ["True image", "Pred image", "True error", "Pred error"]

        fig3, axs = plt.subplots(
            4,
            self.num_img_channels,
            figsize=(2.1 * self.num_img_channels, 7.5),
            constrained_layout=True,
        )

        split = 4  # first 4 channels vs last 2 channels

        for r, row in enumerate(rows):
            # first group
            finite1 = row[:split][np.isfinite(row[:split])]
            if finite1.size > 0:
                vmin1, vmax1 = np.percentile(finite1, [2, 98])
                if np.isclose(vmin1, vmax1):
                    vmax1 = vmin1 + 1e-6
            else:
                vmin1, vmax1 = 0.0, 1.0

            # second group
            finite2 = row[split:][np.isfinite(row[split:])]
            if finite2.size > 0:
                vmin2, vmax2 = np.percentile(finite2, [2, 98])
                if np.isclose(vmin2, vmax2):
                    vmax2 = vmin2 + 1e-6
            else:
                vmin2, vmax2 = 0.0, 1.0

            im1, im2 = None, None

            for c in range(self.num_img_channels):
                ax = axs[r, c]

                if c < split:
                    im = ax.imshow(row[c], cmap="viridis", vmin=vmin1, vmax=vmax1)
                    im1 = im
                else:
                    im = ax.imshow(row[c], cmap="viridis", vmin=vmin2, vmax=vmax2)
                    im2 = im

                if r == 0 or r == 2:
                    patch_alpha = np.kron(
                        mask_grid[c],
                        np.ones((self.img_patch, self.img_patch), dtype=np.float32)
                    )
                    ax.imshow(
                        np.ones_like(patch_alpha),
                        cmap="gray",
                        alpha=patch_alpha * 0.9,
                        vmin=0,
                        vmax=1,
                    )

                if r == 0:
                    ax.set_title(f"ch {c}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(row_labels[r], fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

            # one colorbar for channels 0-3
            cbar1 = fig3.colorbar(im1, ax=axs[r, :split], fraction=0.018, pad=0.01)
            cbar1.ax.tick_params(labelsize=7)

            # one colorbar for channels 4-5
            cbar2 = fig3.colorbar(im2, ax=axs[r, split:], fraction=0.018, pad=0.01)
            cbar2.ax.tick_params(labelsize=7)

        fig3.suptitle("Channel maps", fontsize=10)

        @rank_zero_only
        def log_channel_grid():
            wandb.log({"image_channels": wandb.Image(fig3)})

        log_channel_grid()
        
        plt.show()
        plt.close(fig)
        plt.close(fig2)
        plt.close(fig3)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.warmup:
            return epoch / float(self.warmup)
        return 0.5 * (1. + np.cos(np.pi * (epoch - self.warmup) / (self.max_num_iters - self.warmup)))


if __name__ == "__main__":


    seed_everything(130, workers=True)
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    train_loader, val_loader = CreateMultimodalDataLoadersIter(end=5000, train_size=3500, batch_size=32)
    # train_loader, val_loader = CreateMultimodalDataLoadersIter(end=4737442, train_size=3316209, batch_size=32)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,                         # Save all checkpoints
        every_n_epochs=1,                      
        dirpath=os.path.join(os.environ["SCRATCH"], "DESIMAE/ImageMHP"),  
        filename="{epoch:03d}-{val_loss:.4f}",  # Include val_loss in name
        monitor="val_loss",
        mode="min",
        save_weights_only=False               # Save full model
    )
    
    os.environ["WANDB_DIR"] = os.environ["SCRATCH"]
    os.environ["WANDB_CACHE_DIR"] = os.path.join(os.environ["SCRATCH"], ".cache", "wandb")
    
    wandb.finish()
    
    # logger = WandbLogger(
    #     project="Image-Ablation",
    #     name="Large model, CNN, 500K",
    #     log_model=True,
    # )

    logger = WandbLogger(
        project="Image-Ablation",
        id="nfwlrvvr",
        resume="must",
        log_model=True,
    )
    
    print(f"W&B dashboard: {logger.experiment.url}")
    
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=600,
        logger=logger,
        accelerator="gpu",
        devices="auto",              # <- important with torchrun
        strategy="ddp",
        num_nodes=4,
        precision="32",
        gradient_clip_val=100.0,
        gradient_clip_algorithm="norm",
    )
    
    prob = 0.7/15

    model = MaskedAutoencoderViT(spec_dim=7781,
        max_epochs=600,
        warmup_epoch=5,
        mask_ratio=0.75,
        lam_img_sigma_masked=0.1,
        embed_dim        = 256,
        merged_depth     = 4,
        merged_num_heads = 8,
        s_depth          = 4,
        e_depth          = 4,
        s_num_heads      = 8,
        e_num_heads      = 8,
        decoder_embed_dim= 512,
        decoder_depth    = 8,
        decoder_num_heads= 16,
        decoder_MLP_coefficient = 1,
        patch_scheme={
            "patch_sizes": [1, 1, 2, 4, 8, 16, 32, 64, 128, 64, 32, 16, 8, 4, 2, 1],
            "mask_ratios": [1, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1, 0.0],
            "probs": [0.3, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob]
        })
    
    ckpt_path = "/pscratch/sd/p/pzehao/DESIMAE/ImageMHP/epoch=077-val_loss=-0.5690.ckpt"
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)