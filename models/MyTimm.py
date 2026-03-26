import math
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.layers import (
    Mlp,
    LayerNorm,
    DropPath,
)

def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    """Add attention mask to attention scores if provided."""
    if attn_mask is None:
        return scores
    attn_mask = attn_mask.to(device=scores.device, dtype=scores.dtype)
    return scores + attn_mask

class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = hasattr(F, 'scaled_dot_product_attention')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = None

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.attn_drop(attn)
            self.attn = attn
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class LayerScale(nn.Module):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239
    """

    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        """Initialize LayerScale module.

        Args:
            dim: Dimension.
            init_values: Initial value for scaling.
            inplace: If True, perform inplace operations.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        token_mask_b = None
        if token_mask is not None:
            token_mask_b = token_mask.to(device=x.device, dtype=torch.bool, non_blocking=True)
            x = x.masked_fill(token_mask_b.unsqueeze(0).unsqueeze(-1), 0.0)
            
        attn_result, _ = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.drop_path1(self.ls1(attn_result))
        if token_mask_b is not None:
            x = x.masked_fill(token_mask_b.unsqueeze(0).unsqueeze(-1), 0.0)

        # MLP sublayer
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if token_mask_b is not None:
            x = x.masked_fill(token_mask_b.unsqueeze(0).unsqueeze(-1), 0.0)
        return x

""" Spectrum to Patch Embedding using Conv1d

Modified from TIMM implementation
"""
class PatchEmbed1D(nn.Module):
    """ Spectrum to patch embedding.
    Returns embedded input with shape (spec_dim / patch_size, embed_dim)
    In: x.shape = (B, spec_dim, 1)
    Out: x.shape = (B, num_patches, embed_dim) (+ 1 for CLS token)
    """
    def __init__(self, spec_dim=7781, patch_size=16, embed_dim=16, norm_layer=None):
        super().__init__()
        self.spec_dim = spec_dim
        self.patch_size = patch_size
        self.grid_size = spec_dim // patch_size
        self.num_patches = self.grid_size

        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, S, _ = x.shape
        assert S == self.spec_dim, f"Input spectrum length ({S}) doesn't match model ({self.spec_dim})."
        x = self.proj(x.permute(0, 2, 1)).permute(0, 2, 1) # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x

def generate_attn_mask(patch_size: int,
                       mask_ratio: float,
                       seq_len: int,
                       device=None,
                       dtype=torch.float32):
    """
    Args
    ----
    patch_size : int
        Width (and height) of each band in *token* units.
    mask_ratio : float in (0, 1)
        Fraction of tokens to mask (≈ fraction of bands).
    seq_len : int
        Sequence length L (so the mask is L × L).
    device, dtype : torch parameters (optional).

    Returns
    -------
    attn_mask : (L, L) tensor, float
        0 for visible positions, -inf where *either* the query
        or key token belongs to a masked band.
    token_mask : (L,) bool
        True for tokens that were selected for masking
    """
    patch_size = int(patch_size)       

    # How many contiguous bands of width = patch_size exist in this sequence
    n_bands = math.ceil(seq_len / patch_size)

    n_mask = max(1, int(round(n_bands * mask_ratio)))  # at least one band
    if mask_ratio == 0:
        n_mask = 0
    band_ids = torch.randperm(n_bands, device=device)[:n_mask]

    # Build a 1-D boolean mask over tokens.
    token_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for b in band_ids:
        b = int(b)
        start = b * patch_size
        end   = min((b + 1) * patch_size, seq_len)
        token_mask[start:end] = True

    attn_mask_bool = token_mask.unsqueeze(0) | token_mask.unsqueeze(1)

    attn_mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    attn_mask[attn_mask_bool] = float('-inf')

    # token_mask[0] = False
    # attn_mask[0, 0] = 0
    attn_mask.fill_diagonal_(0)
    # attn_mask[0, :] = 0
    # attn_mask[:, 0] = 0
    # To prevent CLS token from being masked

    return attn_mask, token_mask