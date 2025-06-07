from __future__ import annotations

import glob
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
from finetuning_scheduler import FinetuningScheduler
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)

# MixedPrecisionPlugin
from loguru import logger
from omegaconf import DictConfig

from tlspt.callbacks.final_checkpoint import SaveFinalCheckpoint


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

    start_time = start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{config.experiment_name}_{seed}_{start_time}"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    log_dir = os.path.join(os.getcwd(), "logs")

    logger.info(f"LOCAL_RANK: {local_rank}")
    logger.info(f"RANK: {global_rank}")
    logger.info(f"WORLD_SIZE: {world_size}")

    logger.remove()  # Remove the default handler
    logger.add(sys.stdout, level="INFO")
    logger.add(os.path.join(log_dir, f"{experiment_name}.log"), level="DEBUG")

    logger.info(f"Training model with seed {seed}")
    logger.info(f"Building dataset with seed {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

    cfg_dict = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    tags = config.tags if "tags" in config else []

    wandb_logger = WandbLogger(
        name=experiment_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        tags=tags,
    )

    # Update the W&B config
    wandb_logger.log_hyperparams(cfg_dict)

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
    devices = config.get("devices", 1)
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

    wandb_logger.log_hyperparams(
        {
            "num_centers": config.model.num_centers,
        }
    )

    # Pretrained model
    resume_ckpt = config.get("resume_checkpoint", None)
    if resume_ckpt and config.get("from_checkpoint", None):
        raise ValueError(
            "Cannot resume training from ckpt as well as loading pretrained ckpt"
        )
    elif config.get("from_checkpoint", None):  # Pretrained from ckpt
        wandb_logger.log_hyperparams({"from_checkpoint": config.from_checkpoint})
        logger.info(f"Loading pretrained model from {config.from_checkpoint}")
        backbone = torch.load(config.from_checkpoint, weights_only=False)["state_dict"]
        logger.info(f"Pretrained model loaded")
        model = hydra.utils.instantiate(config.model, backbone=backbone)
    else:  # From scratch or resume
        model = hydra.utils.instantiate(config.model)
        if resume_ckpt:
            logger.info(f"Resuming training from {resume_ckpt}")

    learning_rate = getattr(model, "learning_rate", None)

    wandb_logger.log_hyperparams(
        {
            "learning_rate": learning_rate,
            "batch_size": config.datamodule.batch_size,
            "num_nodes": num_nodes,
            "strategy": strategy,
            "devices": devices,
        }
    )

    # Extra logging for ablations
    # Determine freeze type
    if config.get("model", {}).get("freeze_encoder", False):
        freeze_type = "frozen"
    elif config.get("tune_schedule", None):
        freeze_type = "scheduled"
    elif config.get("from_checkpoint", None):
        freeze_type = "full"
    else:
        freeze_type = "scratch"

    # Extract training percentage
    train_pct = config.datamodule.limit_train_pct

    # Extract checkpoint info
    checkpoint_name = config.get("from_checkpoint", "scratch")
    if checkpoint_name != "scratch":
        checkpoint_name = os.path.basename(checkpoint_name)

    # Log experiment metadata
    wandb_logger.log_hyperparams(
        {
            "ablation/freeze_type": freeze_type,
            "ablation/train_pct": train_pct,
            "ablation/checkpoint": checkpoint_name,
            "ablation/run_index": config.get("run_index", 0),
            "ablation/experiment": "label_efficiency",
        }
    )

    # CALLBACKS=========================================================================================================
    # Val checkpoint callback
    checkpoint_dir = f"checkpoints/{experiment_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Determine if this is a segmentation or pretraining task
    model_class_name = model.__class__.__name__
    is_segmentation = "Segmentation" in model_class_name

    # Set monitoring metric based on task type
    if is_segmentation:
        filename_template = f"best_model_{experiment_name}_ep{{epoch:02d}}_bal_acc{{val/bal_acc_epoch:.4f}}"
        monitor_metric = "val/bal_acc_epoch"
        monitor_mode = "max"
        logger.info("Detected segmentation task, monitoring balanced accuracy")
    else:
        # Pretraining task
        filename_template = (
            f"best_model_{experiment_name}_ep{{epoch:02d}}_loss{{val/loss:.4f}}"
        )
        monitor_metric = "val/loss"
        monitor_mode = "min"
        logger.info("Detected pretraining task, monitoring validation loss")

    # Callback for best model based on validation loss
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        save_weights_only=False,
        filename=filename_template,
        auto_insert_metric_name=False,
    )

    final_checkpoint_callback = SaveFinalCheckpoint(
        dirpath=checkpoint_dir, experiment_name=experiment_name
    )

    lr_monitor = LearningRateMonitor(
        logging_interval="step", log_momentum=False, log_weight_decay=False
    )

    if config.get("model.freeze_encoder", False) and config.get("tune_schedule", None):
        raise ValueError("Cannot freeze encoder and use tune_schedule at the same time")

    if config.get("tune_schedule", None):
        schedule = config.tune_schedule
        wandb_logger.log_hyperparams({"tune_schedule": schedule})
        schedule_callback = FinetuningScheduler(
            ft_schedule=schedule,
            epoch_transitions_only=True,
            restore_best=config.get("restore_best", False),
        )
        callbacks = [
            best_checkpoint_callback,
            final_checkpoint_callback,
            lr_monitor,
            schedule_callback,
        ]
    else:
        callbacks = [best_checkpoint_callback, final_checkpoint_callback, lr_monitor]

    if config.get("no_checkpoint", False):
        logger.info("Not saving checkpoints")
        callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]

    # Extra custom callbacks from config
    # In train.py
    if "callbacks" in config:
        for callback_name, callback_config in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback_config))
    # ==================================================================================================================

    # model = torch.compile(model)

    matmul_precision = config.get("matmul_precision", "high")
    torch.set_float32_matmul_precision(matmul_precision)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.DEBUG)

    gradient_clip_val = config.get("gradient_clip_val", None)
    gradient_clip_algorithm = config.get("gradient_clip_algorithm", "norm")
    if gradient_clip_val is not None:
        logger.info(
            f"Setting gradient clipping: {gradient_clip_algorithm} with value {gradient_clip_val}"
        )
    else:
        logger.info("No gradient clipping applied")

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

    trainer.fit(model, datamodule, ckpt_path=resume_ckpt)

    # Evaluate all checkpoints on test set (only on global rank 0 to avoid duplicates)
    if global_rank == 0:
        logger.info("Evaluating checkpoints on test set...")

        # Search for best checkpoint
        best_checkpoint_pattern = os.path.join(checkpoint_dir, "best_model_*.ckpt")
        best_checkpoint_files = glob.glob(best_checkpoint_pattern)

        best_checkpoint_path = None
        if best_checkpoint_files:
            # If multiple, take the most recent one
            best_checkpoint_path = max(best_checkpoint_files, key=os.path.getmtime)
            logger.info(f"Found best checkpoint: {best_checkpoint_path}")
        else:
            logger.warning(
                f"No best checkpoint found matching pattern: {best_checkpoint_pattern}"
            )

        checkpoint_configs = [
            {
                "name": "first_epoch",
                "path": os.path.join(checkpoint_dir, "first.ckpt"),
                "suffix": "_first",
            },
            {
                "name": "step_97",
                "path": os.path.join(checkpoint_dir, "step_97.ckpt"),
                "suffix": "_step97",
            },
            {
                "name": "last_epoch",
                "path": os.path.join(checkpoint_dir, "last.ckpt"),
                "suffix": "_last",
            },
            {
                "name": "best_model",
                "path": best_checkpoint_path,
                "suffix": "_best",
            },
        ]

        for ckpt_config in checkpoint_configs:
            ckpt_path = ckpt_config["path"]
            ckpt_name = ckpt_config["name"]
            suffix = ckpt_config["suffix"]

            if ckpt_path and os.path.exists(ckpt_path):
                logger.info(f"Evaluating {ckpt_name} checkpoint: {ckpt_path}")

                # Load the checkpoint
                test_model = model.__class__.load_from_checkpoint(
                    ckpt_path,
                    ball_radius=model.ball_radius,
                    scale=model.scale,
                    neighbor_alg=model.neighbor_alg,
                    num_centers=model.num_centers,
                    num_neighbors=model.num_neighbors,
                )

                # Use original trainer but disable logging
                original_logger = trainer.logger
                trainer.logger = False

                test_results = trainer.test(test_model, datamodule, verbose=False)

                # Restore original logger
                trainer.logger = original_logger

                # Log results with test/ prefix and checkpoint suffix
                if test_results and len(test_results) > 0:
                    test_metrics = test_results[0]
                    prefixed_metrics = {}

                    for key, value in test_metrics.items():
                        # Remove 'test/' prefix if it exists, then add our format
                        clean_key = key.replace("test/", "")
                        # Use format: test/metric_checkpoint (e.g., test/loss_first, test/acc_best)
                        prefixed_key = f"test/{clean_key}{suffix}"
                        prefixed_metrics[prefixed_key] = value

                    # Log to wandb
                    wandb_logger.log_metrics(prefixed_metrics)
                    logger.info(f"Logged {ckpt_name} test metrics: {prefixed_metrics}")
                else:
                    logger.warning(f"No test results returned for {ckpt_name}")

            else:
                logger.warning(f"Checkpoint not found: {ckpt_path}")

        logger.info("Test evaluation complete for all checkpoints")


if __name__ == "__main__":
    # Filter out DeepSpeed launcher arguments before Hydra sees them
    filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith("--local_rank")]
    sys.argv[1:] = filtered_args
    main()
