import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from utils.DataProcessing import get_kernel, smooth_data
from utils.AstroImageFunctions import make_rgb


def visualize(model, spec: torch.Tensor, error: torch.Tensor, spec_pred: torch.Tensor,
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
    offset = model.left_patches * model.patch_size
    x_range = np.arange(offset, offset + len(target_smooth))
    ax.plot(x_range, target_smooth, label="Target (smoothed)", color="orange", linewidth=1)
    ax.fill_between(x_range, data_lower, data_upper, color="orange", alpha=0.3, label="±σ data (smoothed)")
    ax.plot(pred_mean, label="Prediction", color="blue")
    ax.fill_between(range(len(pred_mean)), pred_lower, pred_upper, color="blue", alpha=0.3, label="±σ predicted")
    patch_len = model.patch_size
    for j, masked in enumerate(patch_mask_spec):
        if masked:
            start = offset + j * patch_len
            end = offset + (j + 1) * patch_len
            ax.axvspan(start, end, color='red', alpha=0.05)
    ax.set_title(f"Patch Size = {model.patch_size*model.chunk_size}, {model.mask_ratio} (Val Sample {i})")
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

    N = model.patch_embedimg.num_patches
    gh, gw = model.patch_embedimg.grid_size
    H, W = img.shape[-2], img.shape[-1]

    p = pred_img[i].view(model.num_img_channels, N, model.img_patch, model.img_patch)
    p = p.view(model.num_img_channels, gh, gw, model.img_patch, model.img_patch)
    pred_img_formatted = p.permute(0, 1, 3, 2, 4).reshape(model.num_img_channels, H, W).detach().cpu().numpy()

    pe = error_img[i].view(model.num_img_channels, N, model.img_patch, model.img_patch)
    pe = pe.view(model.num_img_channels, gh, gw, model.img_patch, model.img_patch)
    pred_error_formatted = np.exp(pe.permute(0, 1, 3, 2, 4).reshape(model.num_img_channels, H, W).detach().cpu().numpy())

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

    mask_grid = patch_mask_img.float().view(model.num_img_channels, gh, gw).cpu().numpy()
    rows = [img_formatted, pred_img_formatted, img_error_formatted, pred_error_formatted]
    row_labels = ["True image", "Pred image", "True error", "Pred error"]

    fig3, axs = plt.subplots(
        4,
        model.num_img_channels,
        figsize=(2.1 * model.num_img_channels, 7.5),
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

        for c in range(model.num_img_channels):
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
                    np.ones((model.img_patch, model.img_patch), dtype=np.float32)
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
