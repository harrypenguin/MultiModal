"""Main multimodal pretraining entrypoint."""

import os
import sys

sys.path.append("..")

import torch
import wandb
import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.mae import MaskedAutoencoderViT
from utils.data_processing import CreateMultimodalDataLoadersIter

if __name__ == "__main__":
    seed_everything(130, workers=True)

    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            print("Enabled flash SDPA")
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("Enabled memory-efficient SDPA")

    train_loader, val_loader, test_loader = CreateMultimodalDataLoadersIter(
        end=4737442, train_size=4642694, batch_size=32
    )
    # train 98%, val 1%, test 1%

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=1,
        dirpath=os.path.join(os.environ["SCRATCH"], "DESIMAE/ProductionCheckpoints"),
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )

    os.environ["WANDB_DIR"] = os.environ["SCRATCH"]
    os.environ["WANDB_CACHE_DIR"] = os.path.join(
        os.environ["SCRATCH"], ".cache", "wandb"
    )

    wandb.finish()

    logger = WandbLogger(
        project="Production",
        name="Final",
        log_model=True,
    )

    print(f"W&B dashboard: {logger.experiment.url}")

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=200,
        logger=logger,
        accelerator="gpu",
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        num_nodes=4,
        precision="32",
        gradient_clip_val=100.0,
        gradient_clip_algorithm="norm",
    )

    prob = 0.7 / 14
    patch_scheme = {
        "patch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 64, 32, 16, 8, 4, 2, 1],
        "mask_ratios": [
            1.0,
            13 / 14,
            12 / 14,
            11 / 14,
            10 / 14,
            9 / 14,
            8 / 14,
            7 / 14,
            6 / 14,
            5 / 14,
            4 / 14,
            3 / 14,
            2 / 14,
            1 / 14,
            0.0,
        ],
        "probs": [
            0.3,
            prob,
            prob,
            prob,
            prob,
            prob,
            prob,
            prob,
            prob,
            prob,
            prob,
            prob,
            prob,
            prob,
            prob,
        ],
    }

    model = MaskedAutoencoderViT(
        spec_dim=7781,
        max_epochs=200,
        warmup_epoch=5,
        mask_ratio=0.75,
        lam_img_sigma_masked=0.1,
        embed_dim=256,
        merged_depth=4,
        merged_num_heads=8,
        s_depth=4,
        e_depth=4,
        s_num_heads=8,
        e_num_heads=8,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_MLP_coefficient=1,
        patch_scheme=patch_scheme,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
