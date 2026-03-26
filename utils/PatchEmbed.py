""" Spectrum to Patch Embedding using Conv1d

Modified from TIMM implementation
"""
from torch import nn as nn


class PatchEmbed(nn.Module):
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