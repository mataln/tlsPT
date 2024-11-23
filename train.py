from __future__ import annotations

import os
import random
import sys

import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler

# MixedPrecisionPlugin
from loguru import logger
from omegaconf import DictConfig


@hydra.main(
    version_base="1.1",
    config_path="configs/dev/point_mae/",
    config_name="train_hdf5.yaml",
)
def main(config: DictConfig):
    if "seed" in config:
        seed = config.seed
    else:
        seed = 0

    logger.remove()  # Remove the default handler
    logger.add(sys.stdout, level="INFO")

    logger.info(f"Training model with seed {seed}")
    logger.info(f"Building dataset with seed {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    tags = config.tags if "tags" in config else []

    experiment_name = f"{config.experiment_name}_{seed}"

    wandb_logger = WandbLogger(
        name=experiment_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        tags=tags,
    )

    cmd = " ".join(sys.argv)
    wandb_logger.log_hyperparams({"cmd": cmd})
    logger.info(f"Command: {cmd}")

    logger.info(f"Work dir: {os.getcwd()}")

    yaml_str = omegaconf.OmegaConf.to_yaml(config)
    logger.debug(f"Config:\n{yaml_str}")

    dataloader = hydra.utils.instantiate(config.dataloader)

    num_nodes = config.get("num_nodes", 1)
    strategy = config.get("strategy", "ddp")
    devices = config.get("devices", "auto")
    limit_train_batches = config.get("limit_train_batches", 1.0)
    limit_test_batches = config.get("limit_test_batches", 1.0)
    limit_val_batches = config.get("limit_val_batches", 1.0)
    log_every_n_steps = config.get("log_every_n_steps", 1)
    check_val_every_n_epoch = config.get("check_val_every_n_epoch", 1)
    num_sanity_val_steps = config.get("num_sanity_val_steps", 0)
    profiler = config.get("profiler", None)

    if profiler == "advanced":
        logger.info("Using AdvancedProfiler")
        profiler = AdvancedProfiler(
            dirpath="/home/matt/work/tlsPT/profiler", filename="advancedprofiler"
        )
    elif profiler == "simple":
        logger.info("Using SimpleProfiler")
        profiler = SimpleProfiler(
            dirpath="/home/matt/work/tlsPT/profiler", filename="simpleprofiler"
        )
    else:
        logger.info("Not using profiler")

    wandb_logger.log_hyperparams(
        {
            "limit_train_batches": limit_train_batches,
            "limit_test_batches": limit_test_batches,
            "limit_val_batches": limit_val_batches,
        }
    )

    # Val checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last="link",
        save_weights_only=False,
        filename=f"best_model_{experiment_name}_{{epoch:02d}}_{{val/loss:.4f}}",
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback]

    model = hydra.utils.instantiate(config.model)
    # model = torch.compile(model)

    trainer = pl.Trainer(
        num_nodes=num_nodes,
        strategy=strategy,
        devices=devices,
        max_epochs=config.max_epochs,
        log_every_n_steps=log_every_n_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        limit_train_batches=limit_train_batches,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=num_sanity_val_steps,
        profiler=profiler,
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
