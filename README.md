## Repository Structure

```
MultiModal/
├── models/
│   ├── MAE.py                  # MAE model (PyTorch Lightning) with extra loss terms
│   ├── MAESimple.py            # Same model with extra loss terms removed
│   └── MyTimm.py               # Custom transformer blocks (Attention, PatchEmbed1D, attn masks)
│
├── losses/
│   ├── SpecLoss.py             # Loss with extra terms for masked spectra (gradient, curvature, FFT, top-k penalties)
│   └── GaussianNLL.py          # Clean NLL loss, with an extra penalty term for predicted uncertainty (used in the paper)
│
├── utils/
│   ├── DataProcessing.py       # MultimodalDataset (Zarr), dataloaders, distributed sampling, rest-frame indexing
│   ├── Scheduler.py            # Cosine-annealing LR with linear warmup
│   ├── PositionalEmbedding.py  # 1D (spectra) and 2D (images) sinusoidal positional embeddings
│   ├── PatchEmbed.py           # 1D Conv1d patch embedding for spectra
│   ├── AstroImageFunctions.py  # Multi-band flux to RGB conversion (Legacy Survey, WISE, SDSS)
│   └── Visualization.py        # Training/validation visualizations logged to W&B
│
├── train/
│   ├── MaeTrain.py                     # Multi-epoch batch training for model with extra loss terms
│   ├── MaeSimpleTrainOneEpoch.py       # One-epoch-at-a-time training (MAESimple), resumable via chaining
│   ├── MaeTrainOneEpoch.py             # One-epoch-at-a-time training (MAE)
│   ├── FixedCLSTokenTrain.py           # Monolithic training script for an earlier MAE variant
│   ├── eval_halpha_checkpoint_sweep.py # Evaluate checkpoints: predict H-alpha maps, compare to MaNGA truth
│   ├── eval_two_ckpts_vs_baseline.py   # Compare two checkpoints against the linear baseline
│   ├── plot_halpha_from_npy.py         # Render 3-panel truth/prediction/difference PNGs from saved .npy maps
│   ├── DDPModeBench.py                 # DDP training throughput benchmark
│   ├── train.sh                        # SLURM batch job (4 nodes, 10h, MaeTrain.py)
│   ├── simple_chaininteractive.sh      # Loop of 20 interactive 1-epoch jobs (MaeSimpleTrainOneEpoch.py)
│   └── chaininteractive.sh             # Loop of 20 interactive 1-epoch jobs (MaeTrainOneEpoch.py)
│
└── notebooks/
    ├── figure2.ipynb                    # Paper Figure 2: DESI image, MaNGA H-alpha truth, model prediction + spectra
    ├── halpha_zoo.ipynb                 # Gallery of H-alpha predictions on all ~520 DESI-MaNGA cross-matches
    ├── manga_tests.ipynb               # DESI-MaNGA integration demo and tests (Mostly debugging)
    ├── linear_baseline.ipynb            # Ridge-regression baseline: local image patch -> H-alpha
    ├── hypothesis_test_calibration.ipynb # Uncertainty calibration
    ├── plots.ipynb                      # General analysis and plotting (Can ignore)
    └── model_trace.ipynb                # Model architecture tracing (Can ignore, was mostly for debugging)
```

## Data

All data lives on NERSC scratch. The training pipeline expects:

| Asset | Path | Description |
|-------|------|-------------|
| DESI Zarr | `/pscratch/sd/p/pzehao/iron/desi_maglim_19_5.zarr` | 4.7M spectra + images. Arrays: `FLUX` (7781-bin spectra), `IVAR`, `IMG` (128x128, 6 channels), `IMG_IVAR`, `WAVE` |
| Metadata | `/pscratch/sd/p/pzehao/iron/desi_zcat_maglim_19_5.parquet` | Redshift (`Z`), sky coordinates (`TARGET_RA/DEC`, `MEAN_FIBER_RA/DEC`) |
| Cross-match CSV | `/pscratch/sd/p/pzehao/desi_manga_matches.csv` | DESI-MaNGA object pairings for evaluation |
| MaNGA DRP/DAP | `/pscratch/sd/p/pzehao/MyQuota/manga_maps/` | Ground-truth H-alpha maps; auto-downloaded from SDSS DR17 on first use |

## Environment

Perlmutter conda environment:

```
conda activate /global/cfs/cdirs/desi/users/pzehao/envs/peng
```

Key dependencies: `torch`, `pytorch_lightning`, `timm`, `wandb`, `zarr`, `astropy`, `scipy`, `matplotlib`, `pandas`, `joblib`.

## Reproducing Figure 2

### Step 1: Train the model (optional)

From `train/`, launch the batch job:

```bash
sbatch train.sh
```

Note that one epoch takes just over an hour on the full 4-node setup.

Training logs to Weights & Biases (project: `Production`).

### Step 2: Generate H-alpha maps from a checkpoint

Without redoing the pretraining, the current latest checkpoint is located at `/pscratch/sd/p/pzehao/DESIMAE/ProductionCheckpointsSimple/epoch=119-val_loss=-1.5850.ckpt`.
Run the evaluation sweep to produce predicted H-alpha `.npy` files:

```bash
cd train
python eval_halpha_checkpoint_sweep.py \
    --checkpoint-dir $SCRATCH/DESIMAE/ProductionCheckpointsSimple \
    --output-dir ../notebooks/ckpt_halpha_best_only_out \
    --save-maps \
    --max-checkpoints 1
```

This loads the trained model, predicts spectra at each MaNGA spaxel position, extracts H-alpha flux, and saves the resulting map as `.npy`.
This sweep script is the reusable scientific entrypoint for H-alpha map generation; the Figure 2 workflow below just consumes its outputs.

### Step 2b: Reproduce Figure 2 cleanly

Use the dedicated figure workflow:

```bash
python notebooks/figure2_prepare.py \
    --checkpoint /pscratch/sd/p/pzehao/DESIMAE/ProductionCheckpointsSimple/epoch=119-val_loss=-1.5850.ckpt \
    --output-dir notebooks/figure2_cache

python notebooks/figure2_plot.py \
    --cache notebooks/figure2_cache/figure2_cache.pkl \
    --output notebooks/figure2.png
```

The first command does the model/data work once and stores a compact cache. The second command is pure plotting.

### Step 3: Generate the figure

Simply run `notebooks/figure2.ipynb`. It loads the saved H-alpha `.npy`, cross-matches with MaNGA ground truth, and produces figure 2 from our paper.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Multimodal MAE-ViT (separate spectrum/image encoder paths, merged cross-modal blocks, shared decoder) |
| Encoder | embed_dim=256, 4 spectrum blocks, 4 error blocks, 4 merged blocks |
| Decoder | embed_dim=512, depth=8, 16 heads |
| Spectrum input | 7781 wavelength bins, variable patch sizes (1-128) with dynamic masking |
| Image input | 6 channels, 128x128 px, 16x16 patches |
| Optimizer | Adam, lr=2e-4, cosine warmup (1 epoch), gradient clip norm=100 |
| Batch size | 32 per GPU (512 effective across 16 GPUs) |
| Hardware | 4 Perlmutter GPU nodes (16 A100s), DDP, FP32 |
| Data split | 98% train / 1% val / 1% test |
