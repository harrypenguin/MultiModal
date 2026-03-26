import numpy as np
import torch

def get_1d_sincos_pos_embed(embed_dim, seq_len, cls_token=False):
    """
    Generate 1D sine-cosine positional embeddings.

    Args:
        embed_dim: int, output dimension for each position (must be even)
        seq_len: int, number of positions (sequence length)
        cls_token: bool, if True, add a [CLS] token at the beginning

    Returns:
        pos_embed: [seq_len, embed_dim] or [1 + seq_len, embed_dim] if cls_token is True
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    positions = np.arange(seq_len, dtype=float)  # shape (seq_len,)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, positions)

    if cls_token:
        cls_embed = np.zeros((1, embed_dim))
        pos_embed = np.concatenate([cls_embed, pos_embed], axis=0)

    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Args:
        embed_dim: int, embedding dimension
        pos: np.array of shape (M,), positions

    Returns:
        np.array of shape (M, embed_dim), positional embeddings
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / (10000**omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sin-cos PE, got {embed_dim}")

    dim_each = embed_dim // 4
    omega = np.arange(dim_each, dtype=np.float32)
    omega = 1.0 / (10000 ** (omega / dim_each))

    ys, xs = np.meshgrid(
        np.arange(grid_h, dtype=np.float32),
        np.arange(grid_w, dtype=np.float32),
        indexing="ij",
    )
    ys = ys.reshape(-1, 1)
    xs = xs.reshape(-1, 1)

    out_x = xs * omega[None, :]
    out_y = ys * omega[None, :]

    emb_x = np.concatenate([np.sin(out_x), np.cos(out_x)], axis=1)
    emb_y = np.concatenate([np.sin(out_y), np.cos(out_y)], axis=1)
    emb = np.concatenate([emb_x, emb_y], axis=1)
    return torch.from_numpy(emb).float().unsqueeze(0)
