from __future__ import annotations

import os

import lightning.pytorch as pl
from loguru import logger


class SaveFinalCheckpoint(pl.Callback):
    """Saves checkpoints at key training milestones, regardless of performance.

    This callback saves:
    - A checkpoint after the first training epoch completes
    - A final checkpoint when training ends

    Useful for debugging, early stopping analysis, and ensuring you have
    checkpoints even if validation-based checkpointing fails.
    """

    def __init__(self, dirpath, experiment_name):
        self.dirpath = dirpath
        self.experiment_name = experiment_name

    def on_train_epoch_end(self, trainer, pl_module):
        """Save checkpoint after the first training epoch."""
        if (
            trainer.is_global_zero and trainer.current_epoch == 0
        ):  # Only save on rank 0 for DDP and only after first epoch
            trainer.current_epoch
            filepath = os.path.join(self.dirpath, f"first.ckpt")
            trainer.save_checkpoint(filepath)
            logger.info(f"Saved first epoch checkpoint: {filepath}")

    def on_fit_end(self, trainer, pl_module):
        """Save checkpoint when training completes."""
        if trainer.is_global_zero:  # Only save on rank 0 for DDP
            trainer.current_epoch
            filepath = os.path.join(self.dirpath, f"last.ckpt")
            trainer.save_checkpoint(filepath)
            logger.info(f"Saved final checkpoint: {filepath}")
