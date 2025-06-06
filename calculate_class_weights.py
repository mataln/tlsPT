# calculate_class_weights.py
from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig


@hydra.main(config_path="configs/lw_seg/", config_name="vits_scratch_sfp.yaml")
def calculate_weights(config: DictConfig):
    datamodule = hydra.utils.instantiate(config.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    class_counts = torch.zeros(2)  # leaf, wood
    total_points = 0

    for batch in datamodule.train_dataloader():
        labels = batch["features"].squeeze(-1).long()
        class_counts[0] += (labels == 0).sum()
        class_counts[1] += (labels == 1).sum()
        total_points += labels.numel()

    print(
        f"Class 0 (leaf): {class_counts[0]:,} points ({class_counts[0]/total_points*100:.1f}%)"
    )
    print(
        f"Class 1 (wood): {class_counts[1]:,} points ({class_counts[1]/total_points*100:.1f}%)"
    )

    # Inverse frequency weights
    weights = total_points / (2 * class_counts + 1e-6)
    weights = weights / weights.mean()

    print(f"\nClass weights: [{weights[0]:.4f}, {weights[1]:.4f}]")


if __name__ == "__main__":
    calculate_weights()
