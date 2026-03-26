import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split, Subset
from scipy.ndimage import convolve1d
import pandas
import torch
import zarr

def get_extreme_mask(spectra: np.ndarray, ivar: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask where:
    - flux is outside (-100, 100)
    - ivar < 1e-5
    - spectra or ivar are NaN or inf
    """
    return (
        ~np.isfinite(spectra) |
        ~np.isfinite(ivar) |
        (np.abs(spectra) > 100) |
        (ivar < 1e-5)
    )


class MultimodalDataset(Dataset):
    def __init__(self, path, start=0, end=None, augment=False, max_shift=50):
        self.data = zarr.open(path, mode='r')
        self.flux = self.data['FLUX']
        self.ivar = self.data['IVAR']
        self.img = self.data['IMG']
        self.img_ivar = self.data['IMG_IVAR']

        p = pandas.read_parquet(
            '/pscratch/sd/p/pzehao/iron/desi_zcat_maglim_19_5.parquet',
            columns=['Z', 'TARGET_RA', 'TARGET_DEC', 'MEAN_FIBER_RA', 'MEAN_FIBER_DEC'],
        )

        n_total = int(self.flux.shape[0])
        self.start = int(start)
        self.end = int(end) if end is not None else n_total
        self.end = min(self.end, n_total)

        if self.start < 0 or self.start >= self.end:
            raise ValueError(f"Invalid range: start={self.start}, end={self.end}, total={n_total}")

        sl = slice(self.start, self.end)

        self.augment = augment
        self.max_shift = max_shift

        self.redshift = p['Z'].iloc[sl].values.astype(np.float32)
        target_ra = p['TARGET_RA'].iloc[sl].values.astype(np.float32)
        target_dec = p['TARGET_DEC'].iloc[sl].values.astype(np.float32)
        fibre_ra = p['MEAN_FIBER_RA'].iloc[sl].values.astype(np.float32)
        fibre_dec = p['MEAN_FIBER_DEC'].iloc[sl].values.astype(np.float32)

        # Hardcoding values from Biprateep
        pix_scale_arcsec = 0.262
        arcsec_per_deg = 3600.0

        dra_deg = (fibre_ra - target_ra)
        ddec_deg = (fibre_dec - target_dec)
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
            xy_pix = np.array([self.dx_pix[local_idx], self.dy_pix[local_idx]], dtype=np.float32)

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
                spec_tensor.clone(),
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

def CreateMultimodalDataLoadersIter(
    path='/pscratch/sd/p/pzehao/iron/desi_maglim_19_5.zarr',
    end=1000000,
    train_size=700000,
    batch_size=16,
    augment_train=True,
    max_shift=50,
):
    train_base = MultimodalDataset(path, start=0, end=end, augment=augment_train, max_shift=max_shift)
    val_base   = MultimodalDataset(path, start=0, end=end, augment=False, max_shift=0)

    total_size = len(train_base)
    if train_size > total_size:
        raise ValueError(f"train_size ({train_size}) exceeds dataset size ({total_size})")

    val_size = total_size - train_size

    g = torch.Generator().manual_seed(130)
    perm = torch.randperm(total_size, generator=g).tolist()
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_dataset = Subset(train_base, train_idx)
    val_dataset   = Subset(val_base, val_idx)

    num_workers = 7
    loader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=safe_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader

def safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None  # will trigger skip in training loop
    return torch.utils.data.default_collate(batch)

def generate_rest_indices(s: torch.Tensor, z: float, 
                          lambda_min_obs: float = 3600.0,
                          lambda_step_obs: float = 0.8,
                          patch_size: int = 31,
                          lambda_max_rest: float = 10000.0):
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
    patch_indices = torch.arange(num_patches, device=s.device).unsqueeze(0).repeat(B, 1)  # (B, num_patches)

    # Observed-frame wavelengths
    lambda_start_obs = lambda_min_obs + patch_indices * patch_size * lambda_step_obs  # (B, num_patches)
    lambda_end_obs = lambda_start_obs + (patch_size - 1) * lambda_step_obs  # inclusive

    # Convert to rest-frame
    lambda_start_rest = lambda_start_obs / (1 + z)
    lambda_end_rest = lambda_end_obs / (1 + z)

    # Convert to integer indices on rest-frame grid (e.g., [0, 10000))
    rest_start_idx = lambda_start_rest.round().long().clamp(0, int(lambda_max_rest) - 1)
    rest_end_idx = lambda_end_rest.round().long().clamp(0, int(lambda_max_rest) - 1)

    return rest_start_idx, rest_end_idx

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
    kernel = np.exp(-x**2 / (2 * nsmooth**2))
    return kernel

def smooth_data(
    data_in: np.ndarray,
    kernel: np.ndarray,
    ivar_in: np.ndarray = None,
    ivar_weight: bool = False
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
            (data_in * ivar_in) * finite_mask,
            kernel,
            mode='constant',
            cval=0.0
        )
        denominator = convolve1d(
            ivar_in * finite_mask,
            kernel,
            mode='constant',
            cval=0.0
        )
    else:
        # Equivalent to smooth(data) / smooth(ones)
        # The denominator correctly calculates the sum of kernel weights at each point,
        # accounting for edge effects, just like the JS version.
        numerator = convolve1d(
            data_in * finite_mask,
            kernel,
            mode='constant',
            cval=0.0
        )
        denominator = convolve1d(
            finite_mask.astype(float), # convolve with the mask to get the correct weights
            kernel,
            mode='constant',
            cval=0.0
        )

    # To avoid division by zero, set result to 0 where denominator is 0
    smoothed_data = np.zeros_like(data_in)
    np.divide(numerator, denominator, out=smoothed_data, where=denominator!=0)

    return smoothed_data

def smooth_noise(
    noise_in: np.ndarray,
    kernel: np.ndarray,
    ivar_weight: bool = False
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
            noise_in * finite_mask, # noise_in is ivar here
            kernel**2,
            mode='constant',
            cval=0.0
        )
        denominator = convolve1d(
            noise_in * finite_mask,
            kernel,
            mode='constant',
            cval=0.0
        )
        
        # Calculate sqrt(numerator_sq) / denominator
        numerator = np.sqrt(numerator_sq)
    else:
        # Adding noise in quadrature:
        # sigma_smooth^2 = sum(K_i^2 * sigma_i^2) / (sum(K_i))^2
        # We are calculating sigma_smooth.
        numerator_sq = convolve1d(
            (noise_in**2) * finite_mask, # noise_in is sigma here
            kernel**2,
            mode='constant',
            cval=0.0
        )
        denominator = convolve1d(
            finite_mask.astype(float),
            kernel,
            mode='constant',
            cval=0.0
        )
        # Calculate sqrt(numerator_sq) / denominator
        numerator = np.sqrt(numerator_sq)

    smoothed_noise = np.zeros_like(noise_in)
    np.divide(numerator, denominator, out=smoothed_noise, where=denominator!=0)
    
    return smoothed_noise