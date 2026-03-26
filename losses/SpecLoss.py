import torch
import torch.nn.functional as F


def grad1(x):
    # x: (B, L)
    return x[..., 1:] - x[..., :-1]


def grad2(x):
    return x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]


def normalized_weighted_mse(x_hat, x, w, eps=1e-6):
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


def spiky_weight(x, tau=0.8, eps=1e-6):
    """
    Compute a weight that emphasizes high gradients.
    tau: float - quantile threshold for spike detection
    """
    g = grad1(x).abs()
    g = F.pad(g, (1, 0))  # align to length L
    g = g / (g.mean(dim=-1, keepdim=True) + eps)
    thresh = torch.quantile(g, tau, dim=-1, keepdim=True)
    return (g > thresh).float() * g.clamp_min(0)


def topk_mse(x_hat, x, kfrac=0.1):
    """
    Compute the top-k MSE loss.
    kfrac: float - fraction of top-k pixels to consider
    """
    err = (x_hat - x).pow(2)
    L = err.shape[-1]
    k = max(1, int(L * float(kfrac)))
    topk, _ = torch.topk(err, k, dim=-1, largest=True, sorted=False)
    return topk.mean()


def fft_hf_loss(x_hat, x, lam=1.0, eps=1e-6):
    """
    Compute the high-frequency loss using FFT.
    """
    x_hat32 = x_hat.to(torch.float32)
    x32 = x.to(torch.float32)
    Xh = torch.fft.rfft(x_hat32, dim=-1)
    X = torch.fft.rfft(x32, dim=-1)
    K = X.shape[-1]
    k = torch.arange(K, device=x.device, dtype=torch.float32)
    w = (k / k.max().clamp_min(1)).clamp_min(eps).view(1, K)
    return lam * (w * (Xh - X).abs()).mean()


def asym_under_penalty(x_hat, x, lam=1.0):
    """
    Compute an asymmetric penalty for under-predictions.
    This loss penalizes cases where x_hat is less than x.
    """
    under = (x - x_hat).clamp_min(0.0)
    return lam * under.pow(2).mean()


def forward_loss(
    x_hat,
    x,
    w,
    log_s,
    img_hat,
    img,
    weig_img,
    error_img,
    mask,
    img_mask=None,
    *,
    num_patches1d,
    left_patches,
    patch_size,
    img_patch,
    num_img_channels,
    num_img_patches,
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
    P = num_patches1d
    if P is None:
        # Fallback: infer P from patch_size and a known full length
        P = P_in - 1 if P_in * patch_size > L else P_in

    # Drop CLS if present
    if P_in == P + 1:
        patch_mask = mask[..., 1:]
    else:
        patch_mask = mask

    psize = int(max(1, patch_size if patch_size is not None else (L // max(P, 1))))
    # Build full pixel mask over all patches, then crop to center window
    pixel_mask_full = patch_mask.repeat_interleave(psize, dim=-1)  # (B, P*psize)
    # Compute center slice bounds to match x_hat/x
    offset = int(left_patches) * psize
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
        d1 = (grad1(x_hat) - grad1(x)).abs()          # (B, L-1)
        Lg = mask_mean(d1, pixel_mask, shrink='grad1')             # uses (B, L-1) mask
        loss = loss + lam_grad * Lg

    if lam_curv > 0.0:
        d2 = (grad2(x_hat) - grad2(x)).abs()          # (B, L-2)
        Lc = mask_mean(d2, pixel_mask, shrink='grad2')             # uses (B, L-2) mask
        loss = loss + lam_curv * Lc

    if lam_fft > 0.0:
        # Masked residual HF: confine emphasis to masked region
        r = (x_hat - x) * pixel_mask
        loss = loss + fft_hf_loss(r, torch.zeros_like(r), lam=lam_fft)

    if lam_topk > 0.0:
        err = (x_hat - x).pow(2)
        masked_err = torch.where(pixel_mask > 0, err, torch.full_like(err, -float('inf')))
        k = max(1, int(L * float(topk_frac)))
        topk, _ = torch.topk(masked_err, k, dim=-1, largest=True, sorted=False)
        Ltopk = torch.clamp(topk, min=0).mean()
        loss = loss + lam_topk * Ltopk

    if lam_spiky > 0.0:
        W = spiky_weight(x, tau=spiky_tau, eps=eps) * pixel_mask
        loss = loss + lam_spiky * normalized_weighted_mse(x_hat, x, W, eps=eps)

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
    P = img_patch
    C = num_img_channels
    N = num_img_patches
    # (B, 98304) -> (B, C, N, P*P) -> (B, N, C, P*P) -> (B, N, C*P*P)
    img_hat = img_hat.view(B, C, N, P * P).permute(0, 2, 1, 3).reshape(B, N, C * P * P)
    error_img = error_img.view(B, C, N, P * P).permute(0, 2, 1, 3).reshape(B, N, C * P * P)

    weig_img = weig_img.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(img.size(0), N, C * P * P)
    img = img.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(img.size(0), N, C * P * P)

    weig_img = torch.nan_to_num(torch.clamp(weig_img, min=eps), nan=eps, posinf=1.0 / eps, neginf=eps)
    error_img = torch.nan_to_num(error_img, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    img_loss = 0.5 * (((img_hat - img) ** 2) / (1.0 / weig_img + weight * torch.exp(error_img).clamp_min(eps) ** 2)
              + regularizer * torch.log((1.0 / weig_img + weight * torch.exp(error_img).clamp_min(eps) ** 2))).mean()

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
