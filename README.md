## Repository Structure

```
MultiModal/
├── models/
│   ├── mae.py                  # Main model implementation
│   └── mytimm.py               # Custom transformer blocks & related utils
│
├── losses/
│   └── gaussian_nll.py             # Canonical Gaussian NLL loss
│
├── utils/
│   ├── data_processing.py       # Dataset/dataloader objects, distributed sampling, rest-frame indexing, as well as other miscellaneous data processing helpers
│   ├── scheduler.py            # Cosine-annealing LR with linear warmup
│   ├── positional_embedding.py  # 1D (spectra) and 2D (images) sinusoidal positional embeddings
│   ├── patch_embed.py           # 1D Conv1d patch embedding for spectra
│   ├── astro_image_functions.py  # Multi-band flux to RGB conversion (Legacy Survey, WISE, SDSS)
│   └── visualization.py        # Training/validation visualizations for W&B tracking
│
├── pretrain/
│   ├── train.py                     # Main pretraining script
│   └── train.sh                        # SLURM batch job
│
└── inference/
    ├── demo.ipynb                    # Tutorial notebook for basic inference
```

## Data

Our model is trained with data from the Dark Energy Spectroscopic Instrument survey, Data Release 1 ([DESI DR1](https://arxiv.org/abs/2503.14745)). If you have access to NERSC, you may access these assets directly, following the paths below.

| Asset | Path | Description |
|-------|------|-------------|
| DESI Zarr | `/pscratch/sd/p/pzehao/iron/desi_maglim_19_5.zarr` | 4.7M spectra + images. Arrays: `FLUX` (7781-bin spectra), `IVAR`, `IMG` (128x128, 6 channels), `IMG_IVAR`, `WAVE` |
| Metadata | `/pscratch/sd/p/pzehao/iron/desi_zcat_maglim_19_5.parquet` | Redshift (`Z`), sky coordinates (`TARGET_RA/DEC`, `MEAN_FIBER_RA/DEC`) |
| Cross-match CSV | `/pscratch/sd/p/pzehao/desi_manga_matches.csv` | DESI-MaNGA object pairings for evaluation |

## Environment

If you are running on NERSC Perlmutter, you can use the conda environment located at (`/global/cfs/cdirs/desi/users/pzehao/envs/peng`) to reproduce our results. If you are working elsewhere, please ensure that you have the same versions of the dependencies listed in `environment.yml`.

## Pretraining

From `train/`, launch the batch job:

```bash
sbatch train.sh
```

Note that one epoch takes just over an hour on the full 4-node setup. By default, the training logs are saved to Weights & Biases (project: `Production`), so you may need to reconfigure this!

## Inference

`demo.py` is a tutorial notebook that demonstrates how to load the pretrained model and run inference on a small batch of DESI spectra/images. It also shows how to visualize the results.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Spectrum input | 7781 wavelength bins, variable patch sizes (1-128) with dynamic masking |
| Image input | 6 channels, 128x128 px, 16x16 patches |
| Optimizer | Adam, $\text{lr}=2\times10^{-4}$, `CosineWarmupScheduler`, gradient clip norm=100 |
| Batch size | 32 per GPU (512 effective across 16 GPUs) |
| Hardware | 4 Perlmutter GPU nodes (16 A100s), DDP, FP32 |
| Data split | 98% train / 1% val / 1% test |
