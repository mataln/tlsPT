from __future__ import annotations

import os
import random

import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

# MixedPrecisionPlugin
from loguru import logger
from omegaconf import DictConfig


@hydra.main(
    version_base="1.0",
    config_path="configs/dev/toy_model_data/",
    config_name="train.yaml",
)
def main(config: DictConfig):
    if "seed" in config:
        seed = config.seed
    else:
        seed = 0

    logger.info(f"Training model with seed {seed}")
    logger.infof("Building dataset with seed {seed}")

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

    hydra.utils.instantiate(config.dataloader)

    config.get("num_nodes", 1)
    config.get("strategy", "ddp")
    config.get("devices", "auto")
    limit_train_batches = config.get("limit_train_batches", 1.0)
    limit_test_batches = config.get("limit_test_batches", 1.0)
    limit_val_batches = config.get("limit_val_batches", 1.0)

    wandb_logger.log_hyperparams(
        {
            "limit_train_batches": limit_train_batches,
            "limit_test_batches": limit_test_batches,
            "limit_val_batches": limit_val_batches,
        }
    )

    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        strategy=config.get("strategy", "ddp"),
        devices=config.get("devices", "auto"),
        max_epochs=config.max_epochs,
        log_every_n_steps=config.get("log_every_n_steps", 1),
        logger=wandb_logger,
        limit_train_batches=config.get("limit_train_batches", 1.0),
        limit_test_batches=config.get("limit_test_batches", 1.0),
        limit_val_batches=config.get("limit_val_batches", 1.0),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=config.get("num_sanity_val_steps", 0),
    )


if __name__ == "__main__":
    main()
