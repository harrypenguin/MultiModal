# CLAUDE.md

## Project Overview

Multimodal Masked Autoencoder (MAE) for jointly modeling spectroscopic and imaging data from the DESI (Dark Energy Spectroscopic Instrument) survey. Uses a Vision Transformer architecture with separate encoder pathways for spectra and images that merge at higher layers, with uncertainty-aware reconstruction.

- **Spectra**: 7,781 wavelength bins of flux data (0.8 Å resolution)
- **Images**: 6-channel optical/infrared cutouts (128×128 pixels)
- **Metadata**: Redshift (z), fiber positioning (RA/DEC)
- **Data format**: Zarr

## Repository Structure

```
models/          # Neural network architectures
  MAE.py         # Main model — PyTorch Lightning MaskedAutoencoderViT
  MyTimm.py      # Transformer blocks (Attention, Block, PatchEmbed1D)
  RedshiftFlow.py # Conditional flow matching for redshift inference
losses/
  SpecLoss.py    # Spectral reconstruction losses (weighted MSE, gradients, FFT, spiky features)
train/
  MaeTrain.py    # Main training entry point (Lightning Trainer, DDP, W&B)
  FixedCLSTokenTrain.py  # Legacy monolithic training script (reference only)
utils/
  DataProcessing.py      # Zarr dataset, DataLoader creation, preprocessing
  PositionalEmbedding.py # 1D/2D sinusoidal positional encodings (incl. differentiable compute_sincos_pe)
  PatchEmbed.py          # 1D patch embedding layer
  AstroImageFunctions.py # Survey image → RGB conversion
  Visualization.py       # W&B reconstruction plot logging
  Scheduler.py           # LR warmup scheduler
notebooks/
  manga_tests.ipynb      # Experimentation and validation
```

## Dependencies

Python 3 with: PyTorch, PyTorch Lightning, timm, numpy, pandas, scipy, zarr, matplotlib, wandb.

On Perlmutter (NERSC), use the conda environment at:
```
/global/cfs/cdirs/desi/users/pzehao/envs/peng
```

## Training

```bash
python train/MaeTrain.py
```

- Multi-node distributed training (DDP, 4 GPU nodes on Perlmutter)
- Experiment tracking and resumption via Weights & Biases
- Data: 500K DESI samples (350K train / 150K val) from `/pscratch/sd/p/pzehao/iron/desi_maglim_19_5.zarr`
- Checkpoints saved to `${SCRATCH}/DESIMAE/Final/`

Key hyperparameters (configured in `train/MaeTrain.py`):
- Embedding dim: 256–768, decoder dim: 512
- Transformer depth: 4–8 blocks
- LR: 2e-4 with cosine warmup
- 16 masking schemes with variable patch sizes (1–128) and ratios (0.0–0.9)

## Architecture Notes

- **Encoder**: Separate spectral (1D conv patches, size 31) and image (2D patches, 16x16) pathways. Modality-specific blocks process in observed frame, then rest-frame positional embeddings (computed via direct sinusoidal formula, differentiable through z) are applied before merged attention blocks. Image channels are embedded in a single batched call (not per-channel loop).
- **Redshift Inference**: When `z_mask_prob > 0`, a conditional flow matching module (`RedshiftFlow`) infers missing redshifts from pooled observed-frame features (spec CLS + image mean). Uses optimal-transport CFM with Euler integration (50 steps). During training, z is stochastically masked; the flow learns from both direct supervision (flow matching loss on known-z samples) and indirect signal (reconstruction loss backpropagates through differentiable PE to flow params). Loss balancing via learned homoscedastic task weights (Kendall et al. 2018).
- **Positional Embeddings**: Rest-frame spectral PE uses `compute_sincos_pe()` — an analytic sinusoidal function computed directly at continuous wavelengths (no lookup table). This is fully differentiable through z: `PE(lambda_rest) = [sin(lambda_rest * omega), cos(lambda_rest * omega)]` where `lambda_rest = lambda_obs / (1+z)`. Image PE uses pre-computed 2D sinusoidal + channel embeddings cached as buffers.
- **Decoder**: Separate prediction heads for flux, flux error, image, and image error. 2D conv refiners improve image output. Uses same direct sinusoidal PE as encoder for rest-frame spectral tokens.
- **Masking**: CLS token (position 0) is always protected from masking via `has_cls=True` in `generate_attn_mask`. Validation uses fixed masking (`val_patch_size`, `val_mask_ratio`) for deterministic metrics.
- **Losses** (`SpecLoss.py`): Heteroscedastic Gaussian NLL with inverse variance weighting, gradient/curvature penalties, top-k hard-example mining, FFT high-frequency loss, asymmetric under-prediction penalty, spiky-feature emphasis, and sigma regularization on masked patches (both spectral and image). All weights configurable. Weight sanitization via shared `_sanitize_weights`/`_sanitize_log_scale` helpers. Multi-task loss balancing via learned `log_var_spec`, `log_var_img`, `log_var_z` parameters.
- **Efficiency**: Mixed precision training (`precision="16-mixed"`), optional gradient checkpointing (`gradient_checkpointing=True`), flash attention enabled by default, `drop_last=True` on training DataLoader. Single dataset instance shared between train/val splits; augmentation applied via `AugmentedSubset` wrapper on training split only. Image positional embeddings (spatial + channel) are pre-computed and cached as buffers. DDP uses `static_graph=True` with `find_unused_parameters=False`.

## No Tests / Linting / CI

There are no automated tests, linter configs, or CI pipelines. Validation is done via notebook experimentation and W&B metrics.
