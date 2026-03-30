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
losses/
  SpecLoss.py    # Spectral reconstruction losses (weighted MSE, gradients, FFT, spiky features)
train/
  MaeTrain.py    # Main training entry point (Lightning Trainer, DDP, W&B)
  FixedCLSTokenTrain.py  # Legacy monolithic training script (reference only)
utils/
  DataProcessing.py      # Zarr dataset, DataLoader creation, preprocessing
  PositionalEmbedding.py # 1D/2D sinusoidal positional encodings
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

- **Encoder**: Separate spectral (1D conv patches, size 31) and image (2D patches, 16×16) pathways merge via shared attention blocks. Redshift conditions the positional embeddings. Image channels are embedded in a single batched call (not per-channel loop).
- **Decoder**: Separate prediction heads for flux, flux error, image, and image error. 2D conv refiners improve image output.
- **Masking**: CLS token (position 0) is always protected from masking via `has_cls=True` in `generate_attn_mask`. Validation uses fixed masking (`val_patch_size`, `val_mask_ratio`) for deterministic metrics.
- **Losses** (`SpecLoss.py`): Weighted MSE with inverse variance, gradient/curvature penalties, top-k hard-example mining, FFT high-frequency loss, asymmetric under-prediction penalty, and spiky-feature emphasis. All weights configurable. Weight sanitization via shared `_sanitize_weights`/`_sanitize_log_scale` helpers.
- **Efficiency**: Mixed precision training (`precision="16-mixed"`), optional gradient checkpointing (`gradient_checkpointing=True`), flash attention enabled by default, `drop_last=True` on training DataLoader. Single dataset instance shared between train/val splits; augmentation applied via `AugmentedSubset` wrapper on training split only. Image positional embeddings (spatial + channel) are pre-computed and cached as buffers. DDP uses `static_graph=True` with `find_unused_parameters=False`.

## No Tests / Linting / CI

There are no automated tests, linter configs, or CI pipelines. Validation is done via notebook experimentation and W&B metrics.
