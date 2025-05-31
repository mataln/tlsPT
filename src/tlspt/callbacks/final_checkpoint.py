from __future__ import annotations

import os

import lightning.pytorch as pl
from loguru import logger


class SaveFinalCheckpoint(pl.Callback):
    """Saves checkpoint at the end of training, regardless of performance."""

    def __init__(self, dirpath, experiment_name):
        self.dirpath = dirpath
        self.experiment_name = experiment_name

    def on_fit_end(self, trainer, pl_module):
        """Save checkpoint when training completes."""
        if trainer.is_global_zero:  # Only save on rank 0 for DDP
            epoch = trainer.current_epoch
            filepath = os.path.join(
                self.dirpath, f"final_model_{self.experiment_name}_ep{epoch:02d}.ckpt"
            )
            trainer.save_checkpoint(filepath)
            logger.info(f"Saved final checkpoint: {filepath}")
