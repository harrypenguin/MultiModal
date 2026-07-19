"""Dataset and dataloader utilities for multimodal DESI data."""

import numpy as np
import random
import math
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from scipy.ndimage import convolve1d
import pandas
import torch
import zarr


class ContiguousDistributedSampler(Sampler):
    """Shard dataset into contiguous per-rank chunks to preserve index locality."""

    def __init__(self, dataset, drop_last=False):
        self.dataset = dataset
        self.drop_last = bool(drop_last)

    def _rank_world_size(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()), int(
                torch.distributed.get_world_size()
            )
        return 0, 1

    def _num_samples(self, world_size):
        n = len(self.dataset)
        if self.drop_last:
            return n // world_size
        return int(math.ceil(n / world_size))

    def __iter__(self):
        n = len(self.dataset)
        rank, world_size = self._rank_world_size()
        num_samples = self._num_samples(world_size)

        if n == 0 or num_samples == 0:
            return iter([])

        if self.drop_last:
            start = rank * num_samples
            end = start + num_samples
            return iter(range(start, end))

        start = rank * num_samples
        end = min(start + num_samples, n)
        indices = list(range(start, end))

        if len(indices) < num_samples:
            pad = num_samples - len(indices)
            indices.extend((list(range(n)) * ((pad // n) + 1))[:pad])

        return iter(indices)

    def __len__(self):
        _, world_size = self._rank_world_size()
        return self._num_samples(world_size)


def get_extreme_mask(spectra: np.ndarray, ivar: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask where:
    - flux is outside (-100, 100)
    - ivar < 1e-5
    - spectra or ivar are NaN or inf
    """
    return (
        ~np.isfinite(spectra)
        | ~np.isfinite(ivar)
        | (np.abs(spectra) > 100)
        | (ivar < 1e-5)
    )


class MultimodalDataset(Dataset):
    def __init__(self, path, start=0, end=None, augment=False, max_shift=50):
        self.data = zarr.open(path, mode="r")
        self.flux = self.data["FLUX"]
        self.ivar = self.data["IVAR"]
        self.img = self.data["IMG"]
        self.img_ivar = self.data["IMG_IVAR"]

        p = pandas.read_parquet(
            "/pscratch/sd/p/pzehao/iron/desi_zcat_maglim_19_5.parquet",
            columns=["Z", "TARGET_RA", "TARGET_DEC", "MEAN_FIBER_RA", "MEAN_FIBER_DEC"],
        )

        n_total = int(self.flux.shape[0])
        self.start = int(start)
        self.end = int(end) if end is not None else n_total
        self.end = min(self.end, n_total)

        if self.start < 0 or self.start >= self.end:
            raise ValueError(
                f"Invalid range: start={self.start}, end={self.end}, total={n_total}"
            )

        sl = slice(self.start, self.end)

        self.augment = augment
        self.max_shift = max_shift

        self.redshift = p["Z"].iloc[sl].values.astype(np.float32)
        target_ra = p["TARGET_RA"].iloc[sl].values.astype(np.float32)
        target_dec = p["TARGET_DEC"].iloc[sl].values.astype(np.float32)
        fibre_ra = p["MEAN_FIBER_RA"].iloc[sl].values.astype(np.float32)
        fibre_dec = p["MEAN_FIBER_DEC"].iloc[sl].values.astype(np.float32)

        # Hardcoding values from Biprateep
        pix_scale_arcsec = 0.262
        arcsec_per_deg = 3600.0

        dra_deg = fibre_ra - target_ra
        ddec_deg = fibre_dec - target_dec
        dra_deg *= np.cos(np.deg2rad(target_dec))

        dx_arcsec = dra_deg * arcsec_per_deg
        dy_arcsec = ddec_deg * arcsec_per_deg

        # sky-aligned centred coordinates: x=East+, y=North+
        # for north-up/east-left images, East corresponds to negative image-column direction
        ra_to_x_sign = -1.0
        dec_to_y_sign = 1.0

        self.dx_pix = (ra_to_x_sign * dx_arcsec / pix_scale_arcsec).astype(np.float32)
        self.dy_pix = (dec_to_y_sign * dy_arcsec / pix_scale_arcsec).astype(np.float32)

    def _shift_image(self, arr, dx, dy):
        # arr: (C, H, W)
        out = np.roll(arr, shift=dy, axis=-2)  # rows
        out = np.roll(out, shift=dx, axis=-1)  # cols
        return out

    def __getitem__(self, idx):
        try:
            i = self.start + idx
            local_idx = i - self.start

            spectra = np.asarray(self.flux[i], dtype=np.float32)
            ivar = np.asarray(self.ivar[i], dtype=np.float32)
            img = np.asarray(self.img[i], dtype=np.float32)
            img_ivar = np.asarray(self.img_ivar[i], dtype=np.float32)

            extreme_mask = get_extreme_mask(spectra, ivar)
            extreme_mask_img = get_extreme_mask(img, img_ivar)

            if (~extreme_mask).any():
                ivar_mean = np.mean(ivar[~extreme_mask])
                spectra_mean = np.mean(spectra[~extreme_mask])
            else:
                ivar_mean = 1.0
                spectra_mean = 0.0

            if (~extreme_mask_img).any():
                img_ivar_mean = np.mean(img_ivar[~extreme_mask_img])
                img_mean = np.mean(img[~extreme_mask_img])
            else:
                img_ivar_mean = 1.0
                img_mean = 0.0

            ivar[extreme_mask] = ivar_mean
            spectra[extreme_mask] = spectra_mean
            img_ivar[extreme_mask_img] = img_ivar_mean
            img[extreme_mask_img] = img_mean

            error = 1.0 / np.sqrt(ivar + 1e-6)
            img_error = 1.0 / np.sqrt(img_ivar + 1e-6)

            z = np.float32(self.redshift[local_idx])
            xy_pix = np.array(
                [self.dx_pix[local_idx], self.dy_pix[local_idx]], dtype=np.float32
            )

            if self.augment and self.max_shift > 0:
                dx = np.random.randint(-self.max_shift, self.max_shift + 1)
                dy = np.random.randint(-self.max_shift, self.max_shift + 1)

                img = self._shift_image(img, dx, dy)
                img_ivar = self._shift_image(img_ivar, dx, dy)
                img_error = self._shift_image(img_error, dx, dy)

                xy_pix = xy_pix + np.array([dx, -dy], dtype=np.float32)

            spec_tensor = torch.from_numpy(spectra)
            return (
                spec_tensor,
                spec_tensor,
                torch.from_numpy(ivar),
                torch.from_numpy(error),
                torch.from_numpy(img),
                torch.from_numpy(img_ivar),
                torch.from_numpy(img_error),
                torch.tensor(z, dtype=torch.float32),
                torch.from_numpy(xy_pix),
            )

        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
            return None

    def __len__(self):
        return self.end - self.start


class AugmentedSubset(Dataset):
    def __init__(self, subset: Subset, max_shift=50):
        self.subset = subset
        self.max_shift = max_shift

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        if sample is None:
            return None

        if self.max_shift <= 0:
            return sample

        dx = int(np.random.randint(-self.max_shift, self.max_shift + 1))
        dy = int(np.random.randint(-self.max_shift, self.max_shift + 1))

        x, spec, ivar, error, img, img_ivar, img_error, z, xy_pix = sample
        img = torch.roll(img, shifts=(dy, dx), dims=(-2, -1))
        img_ivar = torch.roll(img_ivar, shifts=(dy, dx), dims=(-2, -1))
        img_error = torch.roll(img_error, shifts=(dy, dx), dims=(-2, -1))
        xy_pix = xy_pix + torch.tensor([dx, -dy], dtype=xy_pix.dtype)

        return x, spec, ivar, error, img, img_ivar, img_error, z, xy_pix


def CreateMultimodalDataLoadersIter(
    path="/pscratch/sd/p/pzehao/iron/desi_maglim_19_5.zarr",
    end=1000000,
    train_size=700000,
    val_size=None,
    test_size=None,
    batch_size=16,
    augment_train=True,
    max_shift=50,
    train_index_mode="random",
    train_block_size=100,
    train_interleave_groups=4,
    train_interleave_span=1,
    shuffle_train=None,
    train_subset_size=None,
    train_cycle_epoch=0,
    train_cycle_drop_last=False,
    num_workers=7,
    prefetch_factor=4,
    distributed_shard_mode="lightning",
):
    base_dataset = MultimodalDataset(path, start=0, end=end, augment=False, max_shift=0)

    total_size = len(base_dataset)
    if train_size > total_size:
        raise ValueError(
            f"train_size ({train_size}) exceeds dataset size ({total_size})"
        )

    remaining = total_size - train_size
    if val_size is None and test_size is None:
        val_size = remaining // 2
        test_size = remaining - val_size
    elif val_size is None:
        val_size = remaining - int(test_size)
    elif test_size is None:
        test_size = remaining - int(val_size)

    val_size = int(val_size)
    test_size = int(test_size)

    if val_size < 0 or test_size < 0:
        raise ValueError(
            f"val_size and test_size must be non-negative (got val_size={val_size}, test_size={test_size})"
        )
    if train_size + val_size + test_size != total_size:
        raise ValueError(
            f"Split sizes must sum to dataset size ({total_size}), got "
            f"train={train_size}, val={val_size}, test={test_size}"
        )

    g = torch.Generator().manual_seed(130)
    perm = torch.randperm(total_size, generator=g).tolist()
    train_idx = perm[:train_size]

    def _interleave_blocks(blocks, group_count, span):
        """Interleave nearby blocks to improve mixing while preserving locality."""
        if not blocks:
            return []

        interleaved = []
        for group_start in range(0, len(blocks), group_count):
            group = blocks[group_start : group_start + group_count]
            max_len = max(len(block) for block in group)
            for offset in range(0, max_len, span):
                for block in group:
                    if offset < len(block):
                        interleaved.extend(block[offset : offset + span])
        return interleaved

    if train_index_mode == "random":
        pass
    elif train_index_mode in ("block_shuffle", "interleave"):
        if train_block_size <= 0:
            raise ValueError(f"train_block_size must be > 0 (got {train_block_size})")

        if train_index_mode == "interleave":
            if train_interleave_groups <= 0:
                raise ValueError(
                    f"train_interleave_groups must be > 0 (got {train_interleave_groups})"
                )
            if train_interleave_span <= 0:
                raise ValueError(
                    f"train_interleave_span must be > 0 (got {train_interleave_span})"
                )

        # Improve Zarr locality: keep contiguous indices within blocks while
        # still randomizing block order to preserve sample mixing.
        train_idx = sorted(train_idx)
        blocks = [
            train_idx[i : i + train_block_size]
            for i in range(0, len(train_idx), train_block_size)
        ]
        py_rng = random.Random(130)
        py_rng.shuffle(blocks)

        if train_index_mode == "block_shuffle":
            train_idx = [idx for block in blocks for idx in block]
        else:
            train_idx = _interleave_blocks(
                blocks,
                group_count=train_interleave_groups,
                span=train_interleave_span,
            )
    else:
        raise ValueError(
            f"Unsupported train_index_mode '{train_index_mode}'. "
            "Expected one of: ['random', 'block_shuffle', 'interleave']"
        )

    if train_subset_size is not None:
        train_subset_size = int(train_subset_size)

    if (
        train_subset_size is not None
        and train_subset_size > 0
        and train_subset_size < len(train_idx)
    ):
        cycle_epoch = int(train_cycle_epoch)
        cycle_epoch = max(0, cycle_epoch)

        # Rotate subset-window order by epoch while keeping full-epoch coverage.
        windows = [
            train_idx[i : i + train_subset_size]
            for i in range(0, len(train_idx), train_subset_size)
        ]

        if train_cycle_drop_last and windows and len(windows[-1]) < train_subset_size:
            windows = windows[:-1]

        if not windows:
            raise ValueError(
                f"train_subset_size={train_subset_size} is too large for train_idx size={len(train_idx)}"
            )

        start_window = cycle_epoch % len(windows)
        ordered_windows = windows[start_window:] + windows[:start_window]
        train_idx = [idx for window in ordered_windows for idx in window]

    val_start = train_size
    val_end = val_start + val_size
    val_idx = perm[val_start:val_end]
    test_idx = perm[val_end:]

    train_subset = Subset(base_dataset, train_idx)
    train_dataset = (
        AugmentedSubset(train_subset, max_shift=max_shift)
        if augment_train
        else train_subset
    )
    val_dataset = Subset(base_dataset, val_idx)
    test_dataset = Subset(base_dataset, test_idx)

    if distributed_shard_mode not in ("lightning", "contiguous"):
        raise ValueError(
            f"Unsupported distributed_shard_mode '{distributed_shard_mode}'. "
            "Expected one of: ['lightning', 'contiguous']"
        )

    num_workers = int(num_workers)
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0 (got {num_workers})")

    loader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=safe_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    if shuffle_train is None:
        # For locality-aware index order, avoid a second full shuffle in the loader.
        subset_active = (
            train_subset_size is not None
            and train_subset_size > 0
            and train_subset_size < train_size
        )
        if subset_active:
            shuffle_train = False
        else:
            shuffle_train = train_index_mode == "random"

    train_sampler = None
    val_sampler = None
    test_sampler = None
    if distributed_shard_mode == "contiguous":
        train_sampler = ContiguousDistributedSampler(train_dataset, drop_last=False)
        val_sampler = ContiguousDistributedSampler(val_dataset, drop_last=False)
        test_sampler = ContiguousDistributedSampler(test_dataset, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(shuffle_train if train_sampler is None else False),
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None  # will trigger skip in training loop
    return torch.utils.data.default_collate(batch)


def generate_rest_indices(
    s: torch.Tensor,
    z: float,
    lambda_min_obs: float = 3600.0,
    lambda_step_obs: float = 0.8,
    patch_size: int = 31,
    lambda_max_rest: float = 10000.0,
):
    """
    Generates rest-frame start and end indices for each spectral patch.

    Parameters:
    - s: Tensor of shape (B, num_patches, embed_dim), patch-embedded spectra
    - z: Redshift (scalar float)
    - lambda_min_obs: Minimum observed wavelength (default = 3600 Å)
    - lambda_step_obs: Wavelength step size in observed frame (default = 1 Å)
    - patch_size: Number of wavelength bins per patch
    - lambda_max_rest: Maximum rest-frame wavelength to clip to (default = 10000 Å)

    Returns:
    - rest_start_idx: (B, num_patches) tensor of start indices into PE
    - rest_end_idx: (B, num_patches) tensor of end indices into PE
    """

    B, num_patches, _ = s.shape
    z = z.unsqueeze(1)

    # Patch indices: 0 to num_patches - 1
    patch_indices = (
        torch.arange(num_patches, device=s.device).unsqueeze(0).repeat(B, 1)
    )  # (B, num_patches)

    # Observed-frame wavelengths
    lambda_start_obs = (
        lambda_min_obs + patch_indices * patch_size * lambda_step_obs
    )  # (B, num_patches)
    lambda_end_obs = lambda_start_obs + (patch_size - 1) * lambda_step_obs  # inclusive

    # Convert to rest-frame
    lambda_start_rest = lambda_start_obs / (1 + z)
    lambda_end_rest = lambda_end_obs / (1 + z)

    # Convert to integer indices on rest-frame grid (e.g., [0, 10000))
    rest_start_idx = lambda_start_rest.round().long().clamp(0, int(lambda_max_rest) - 1)
    rest_end_idx = lambda_end_rest.round().long().clamp(0, int(lambda_max_rest) - 1)

    return rest_start_idx, rest_end_idx


# Spectra smoothing utils from Biprateep
def get_kernel(nsmooth: int) -> np.ndarray:
    """
    Generates a Gaussian kernel for smoothing.

    This is a Python/NumPy equivalent of the get_kernel JavaScript function.

    Args:
        nsmooth: The standard deviation (sigma) of the Gaussian kernel in pixels.

    Returns:
        A 1D NumPy array containing the kernel values.
    """
    if nsmooth <= 0:
        return np.array([])
    # The kernel extends to 2*nsmooth on each side of the center, matching the JS implementation.
    x = np.arange(-2 * nsmooth, 2 * nsmooth + 1)
    kernel = np.exp(-(x**2) / (2 * nsmooth**2))
    return kernel


def smooth_data(
    data_in: np.ndarray,
    kernel: np.ndarray,
    ivar_in: np.ndarray = None,
    ivar_weight: bool = False,
) -> np.ndarray:
    """
    Smooths data using a provided kernel, with optional inverse variance weighting.

    This function vectorizes the logic from the original JavaScript `smooth_data`
    using `scipy.ndimage.convolve1d` for performance and accuracy, especially
    at the boundaries.

    Args:
        data_in: The input data array (e.g., flux).
        kernel: The smoothing kernel.
        ivar_in: The inverse variance array for weighting. Required if ivar_weight is True.
        ivar_weight: If True, apply inverse variance weighting.

    Returns:
        The smoothed data array.
    """
    if kernel.size == 0 or data_in.size == 0:
        return np.copy(data_in)

    # The JS code checks for finite values inside the loop. We can do this upfront
    # by creating a mask and zeroing out non-finite values.
    finite_mask = np.isfinite(data_in)
    if ivar_weight:
        if ivar_in is None:
            raise ValueError("ivar_in must be provided when ivar_weight is True.")
        if ivar_in.shape != data_in.shape:
            raise ValueError("ivar_in must have the same shape as data_in.")
        finite_mask &= np.isfinite(ivar_in)

    # Use a convolution operation, which is equivalent to the nested loops in JS.
    # The `convolve1d` function from SciPy handles boundary conditions gracefully.
    # `mode='constant'` with `cval=0` mimics the JS behavior of ignoring out-of-bounds pixels.

    if ivar_weight:
        # Equivalent to smooth(data*ivar) / smooth(ivar)
        # We multiply by the finite_mask to zero out non-finite values before convolution.
        numerator = convolve1d(
            (data_in * ivar_in) * finite_mask, kernel, mode="constant", cval=0.0
        )
        denominator = convolve1d(
            ivar_in * finite_mask, kernel, mode="constant", cval=0.0
        )
    else:
        # Equivalent to smooth(data) / smooth(ones)
        # The denominator correctly calculates the sum of kernel weights at each point,
        # accounting for edge effects, just like the JS version.
        numerator = convolve1d(data_in * finite_mask, kernel, mode="constant", cval=0.0)
        denominator = convolve1d(
            finite_mask.astype(
                float
            ),  # convolve with the mask to get the correct weights
            kernel,
            mode="constant",
            cval=0.0,
        )

    # To avoid division by zero, set result to 0 where denominator is 0
    smoothed_data = np.zeros_like(data_in)
    np.divide(numerator, denominator, out=smoothed_data, where=denominator != 0)

    return smoothed_data


def smooth_noise(
    noise_in: np.ndarray, kernel: np.ndarray, ivar_weight: bool = False
) -> np.ndarray:
    """
    Smooths noise or ivar using a provided kernel, propagating errors correctly.

    This function vectorizes the logic from the original JavaScript `smooth_noise`.

    Args:
        noise_in: The input noise (stddev) or inverse variance (ivar) array.
        kernel: The smoothing kernel.
        ivar_weight: If True, `noise_in` is treated as ivar, and the error
                     propagation for a weighted mean is used. Otherwise, it's
                     treated as noise to be added in quadrature.

    Returns:
        The smoothed noise or ivar array.
    """
    if kernel.size == 0 or noise_in.size == 0:
        return np.copy(noise_in)

    finite_mask = np.isfinite(noise_in)

    if ivar_weight:
        # Propagating error for a weighted mean:
        # sigma_smooth^2 = sum(K_i^2 * ivar_i) / (sum(K_i * ivar_i))^2
        # We are calculating sigma_smooth.
        numerator_sq = convolve1d(
            noise_in * finite_mask,  # noise_in is ivar here
            kernel**2,
            mode="constant",
            cval=0.0,
        )
        denominator = convolve1d(
            noise_in * finite_mask, kernel, mode="constant", cval=0.0
        )

        # Calculate sqrt(numerator_sq) / denominator
        numerator = np.sqrt(numerator_sq)
    else:
        # Adding noise in quadrature:
        # sigma_smooth^2 = sum(K_i^2 * sigma_i^2) / (sum(K_i))^2
        # We are calculating sigma_smooth.
        numerator_sq = convolve1d(
            (noise_in**2) * finite_mask,  # noise_in is sigma here
            kernel**2,
            mode="constant",
            cval=0.0,
        )
        denominator = convolve1d(
            finite_mask.astype(float), kernel, mode="constant", cval=0.0
        )
        # Calculate sqrt(numerator_sq) / denominator
        numerator = np.sqrt(numerator_sq)

    smoothed_noise = np.zeros_like(noise_in)
    np.divide(numerator, denominator, out=smoothed_noise, where=denominator != 0)

    return smoothed_noise
