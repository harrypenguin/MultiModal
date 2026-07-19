"""Gaussian negative log-likelihood loss helpers."""

import torch


def forward_loss(
    x_hat,
    x,
    w,
    log_s,
    img_hat,
    img,
    weig_img,
    error_img,
    spec_mask=None,
    img_mask=None,
    *,
    img_patch,
    num_img_channels,
    num_img_patches,
    weight=1.0,
    regularizer=1.0,
    eps=1e-6,
    lam_img_sigma_masked=0.0,
):
    # Base spectrum loss over all pixels.
    w_safe = torch.nan_to_num(
        torch.clamp(w, min=eps), nan=eps, posinf=1.0 / eps, neginf=eps
    )
    log_s = torch.nan_to_num(log_s, nan=0.0, posinf=10.0, neginf=-10.0).clamp(
        -10.0, 10.0
    )
    sigma_hat_sq = weight * torch.exp(log_s).clamp(min=eps).pow(2)
    denom = (1.0 / w_safe + sigma_hat_sq).clamp(min=eps)

    sq_error = (x_hat - x).pow(2)
    base_pixel = 0.5 * (
        sq_error / denom + regularizer * torch.log(denom.clamp_min(eps))
    )
    loss = base_pixel.mean()

    if lam_img_sigma_masked > 0.0 and spec_mask is not None:
        if spec_mask.dim() == 1:
            spec_mask = spec_mask.unsqueeze(0).expand(x_hat.size(0), -1)

        target_len = log_s.shape[-1]
        if spec_mask.shape[-1] < target_len:
            spec_mask = torch.nn.functional.pad(
                spec_mask, (0, target_len - spec_mask.shape[-1])
            )
        elif spec_mask.shape[-1] > target_len:
            spec_mask = spec_mask[..., :target_len]

        spec_mask = spec_mask.to(device=log_s.device, dtype=log_s.dtype)
        sigma_spec = torch.exp(log_s).clamp_min(eps)
        denom_spec = spec_mask.sum().clamp_min(1.0)
        spec_sigma_penalty = (sigma_spec.pow(2) * spec_mask).sum() / denom_spec
        loss = loss + lam_img_sigma_masked * spec_sigma_penalty

    B = img.size(0)
    P = img_patch
    C = num_img_channels
    N = num_img_patches
    # (B, 98304) -> (B, C, N, P*P) -> (B, N, C, P*P) -> (B, N, C*P*P)
    img_hat = img_hat.view(B, C, N, P * P).permute(0, 2, 1, 3).reshape(B, N, C * P * P)
    error_img = (
        error_img.view(B, C, N, P * P).permute(0, 2, 1, 3).reshape(B, N, C * P * P)
    )

    weig_img = (
        weig_img.unfold(2, P, P)
        .unfold(3, P, P)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(img.size(0), N, C * P * P)
    )
    img = (
        img.unfold(2, P, P)
        .unfold(3, P, P)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(img.size(0), N, C * P * P)
    )

    weig_img = torch.nan_to_num(
        torch.clamp(weig_img, min=eps), nan=eps, posinf=1.0 / eps, neginf=eps
    )
    error_img = torch.nan_to_num(error_img, nan=0.0, posinf=10.0, neginf=-10.0).clamp(
        -10.0, 10.0
    )

    img_loss = (
        0.5
        * (
            ((img_hat - img) ** 2)
            / (1.0 / weig_img + weight * torch.exp(error_img).clamp_min(eps) ** 2)
            + regularizer
            * torch.log(
                (1.0 / weig_img + weight * torch.exp(error_img).clamp_min(eps) ** 2)
            )
        ).mean()
    )

    if lam_img_sigma_masked > 0.0 and img_mask is not None:
        if img_mask.dim() == 1:
            img_mask = img_mask.unsqueeze(0).expand(B, -1)

        img_mask = img_mask.to(
            device=error_img.device, dtype=error_img.dtype
        ).unsqueeze(-1)

        error_img_tok = (
            error_img.view(B, N, C, P * P).permute(0, 2, 1, 3).reshape(B, C * N, P * P)
        )
        sigma_img = torch.exp(error_img_tok).clamp_min(eps)

        denom = (img_mask.sum() * (P * P)).clamp_min(1.0)
        img_sigma_penalty = (sigma_img.pow(2) * img_mask).sum() / denom
        img_loss = img_loss + lam_img_sigma_masked * img_sigma_penalty

    return loss, img_loss, loss + img_loss
