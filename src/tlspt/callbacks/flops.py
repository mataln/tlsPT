from __future__ import annotations

import lightning.pytorch as pl
import torch
from fvcore.nn import FlopCountAnalysis


class FLOPsCallback(pl.Callback):
    def __init__(self):
        self.forward_flops_per_batch = None
        self.backward_flops_per_param = None
        self.total_forward_flops = 0
        self.total_backward_flops = 0
        self.epoch_forward_flops = 0
        self.epoch_backward_flops = 0

    def on_train_start(self, trainer, pl_module):
        # Get a real batch from the dataloader
        dataloader = trainer.train_dataloader
        batch = next(iter(dataloader))

        # Move batch to the correct device
        device = pl_module.device
        if isinstance(batch, dict):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }

        # Calculate forward FLOPs for the batch
        batch_size = batch["points"].shape[0]
        total_flops = FlopCountAnalysis(pl_module, batch).total()
        self.forward_flops_per_batch = total_flops / batch_size

        # Rough backward estimate: ~2x forward FLOPs for all params
        total_params = sum(p.numel() for p in pl_module.parameters())
        self.backward_flops_per_param = 2 * self.forward_flops_per_batch / total_params

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset epoch counters
        self.epoch_forward_flops = 0
        self.epoch_backward_flops = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Accumulate forward FLOPs
        self.epoch_forward_flops += self.forward_flops_per_batch
        self.total_forward_flops += self.forward_flops_per_batch

        # Calculate and accumulate backward FLOPs based on current trainable params
        trainable_params = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )
        backward_flops = trainable_params * self.backward_flops_per_param
        self.epoch_backward_flops += backward_flops
        self.total_backward_flops += backward_flops

    def on_train_epoch_end(self, trainer, pl_module):
        # Log per-epoch FLOPs
        trainer.logger.log_metrics(
            {
                "flops/epoch_forward": self.epoch_forward_flops,
                "flops/epoch_backward": self.epoch_backward_flops,
                "flops/epoch_total": self.epoch_forward_flops
                + self.epoch_backward_flops,
                "flops/trainable_params": sum(
                    p.numel() for p in pl_module.parameters() if p.requires_grad
                ),
            }
        )

    def on_fit_end(self, trainer, pl_module):
        # Log total FLOPs for entire training
        trainer.logger.log_metrics(
            {
                "flops/total_forward": self.total_forward_flops,
                "flops/total_backward": self.total_backward_flops,
                "flops/total_training": self.total_forward_flops
                + self.total_backward_flops,
            }
        )
