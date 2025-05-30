from __future__ import annotations

import time

import lightning.pytorch as pl


class TrainingTimeCallback(pl.Callback):
    def __init__(self):
        self.train_time_seconds = 0
        self.batch_start = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_time_seconds += time.time() - self.batch_start

    def on_fit_end(self, trainer, pl_module):
        trainer.logger.log_metrics({"train_time_hours": self.train_time_seconds / 3600})
