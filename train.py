from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime

import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)

# MixedPrecisionPlugin
from loguru import logger
from omegaconf import DictConfig

import wandb


@hydra.main(
    version_base="1.1",
    config_path="configs/dev/point_mae/cluster/lw_seg/",
    config_name="DEVTEST.yaml",
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
    pl.seed_everything(seed)

    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    tags = config.tags if "tags" in config else []

    start_time = start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{config.experiment_name}_{seed}_{start_time}"

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

    datamodule = hydra.utils.instantiate(config.datamodule)

    # Length of train, val, test dataloaders
    logger.info(f"Train dataloader length: {len(datamodule.train_dataloader())}")
    logger.info(f"Val dataloader length: {len(datamodule.val_dataloader())}")
    logger.info(f"Test dataloader length: {len(datamodule.test_dataloader())}")

    num_nodes = config.get("num_nodes", 1)
    strategy = config.get("strategy", "ddp")
    devices = config.get("devices", "auto")
    limit_train_pct = config.get("limit_train_pct", 1.0)
    limit_test_pct = config.get("limit_test_pct", 1.0)
    limit_val_pct = config.get("limit_val_pct", 1.0)
    log_every_n_steps = config.get("log_every_n_steps", 1)
    check_val_every_n_epoch = config.get("check_val_every_n_epoch", 1)
    num_sanity_val_steps = config.get("num_sanity_val_steps", 0)
    profiler = config.get("profiler", None)

    if profiler == "advanced":
        logger.info("Using AdvancedProfiler")
        profiler = AdvancedProfiler(dirpath="profiler/", filename="advancedprofiler")
    elif profiler == "simple":
        logger.info("Using SimpleProfiler")
        profiler = SimpleProfiler(dirpath="profiler/", filename="simpleprofiler")
    elif profiler == "pytorch":
        logger.info("Using PyTorch Profiler")
        profiler = PyTorchProfiler()
    else:
        logger.info("Not using profiler")

    wandb_logger.log_hyperparams(
        {
            "limit_train_pct": limit_train_pct,
            "limit_test_pct": limit_test_pct,
            "limit_val_pct": limit_val_pct,
        }
    )

    # Val checkpoint callback
    checkpoint_dir = f"checkpoints/{experiment_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last="link",
        save_weights_only=False,
        filename=f"best_model_{experiment_name}_ep{{epoch:02d}}_loss{{val/loss:.4f}}",
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback]

    model = hydra.utils.instantiate(config.model)
    # model = torch.compile(model)

    matmul_precision = config.get("matmul_precision", "high")
    torch.set_float32_matmul_precision(matmul_precision)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.DEBUG)
    trainer = pl.Trainer(
        num_nodes=num_nodes,
        strategy=strategy,
        devices=devices,
        max_epochs=config.max_epochs,
        log_every_n_steps=log_every_n_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=num_sanity_val_steps,
        profiler=profiler,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
