import os
import sys

sys.path.append("..")

import torch
import wandb
import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from models.MAE import MaskedAutoencoderViT
from utils.DataProcessing import CreateMultimodalDataLoadersIter


if __name__ == "__main__":
    seed_everything(130, workers=True)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    train_loader, val_loader = CreateMultimodalDataLoadersIter(end=500000, train_size=350000, batch_size=32)
    # train_loader, val_loader = CreateMultimodalDataLoadersIter(end=4737442, train_size=3316209, batch_size=32)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        every_n_epochs=5,
        dirpath=os.path.join(os.environ["SCRATCH"], "DESIMAE/Final"),
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )

    os.environ["WANDB_DIR"] = os.environ["SCRATCH"]
    os.environ["WANDB_CACHE_DIR"] = os.path.join(os.environ["SCRATCH"], ".cache", "wandb")

    wandb.finish()

    # logger = WandbLogger(
    #     project="Image-Ablation",
    #     name="Large model, CNN, 500K",
    #     log_model=True,
    # )

    logger = WandbLogger(
        project="Image-Ablation",
        id="nfwlrvvr",
        resume="must",
        log_model=True,
    )

    print(f"W&B dashboard: {logger.experiment.url}")

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Enable flash/memory-efficient attention backends when available
    # (no-op on hardware that doesn't support them)
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=600,
        logger=logger,
        accelerator="gpu",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        num_nodes=4,
        precision="16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    patch_sizes =  [1, 1, 2, 4, 8, 16, 32, 64, 128, 64, 32, 16, 8, 4, 2, 1]
    mask_ratios =  [1, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1, 0.0]
    # First scheme (full masking) gets 30% probability; the rest share 70% equally
    prob_first = 0.3
    prob_rest = (1.0 - prob_first) / (len(patch_sizes) - 1)
    probs = [prob_first] + [prob_rest] * (len(patch_sizes) - 1)

    model = MaskedAutoencoderViT(
        spec_dim=7781,
        max_epochs=600,
        warmup_epoch=5,
        mask_ratio=0.75,
        lam_img_sigma_masked=0.1,
        lam_spec_sigma_masked=0.1,
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
        patch_scheme={
            "patch_sizes": patch_sizes,
            "mask_ratios": mask_ratios,
            "probs": probs,
        },
        z_mask_prob=0.3,
        z_flow_hidden_dim=256,
        z_flow_num_layers=4,
        z_flow_steps=50,
    )

    ckpt_path = "/pscratch/sd/p/pzehao/DESIMAE/ImageMHP/epoch=077-val_loss=-0.5690.ckpt"
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
