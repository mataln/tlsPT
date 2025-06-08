# analyze_dataset.py
from __future__ import annotations

import torch
from tqdm import tqdm

from tlspt.datamodules.components.octree_dataset import OctreeDataset


def analyze_full_dataset(split_file, split, scale):
    dataset = OctreeDataset(
        split_file=split_file,
        split=split,
        scale=scale,
        feature_names=["scalar_truth"],
        normalize=False,
        transform=None,
    )

    print(f"\n=== Analyzing {split} dataset (ALL {len(dataset)} items) ===")

    total_points = 0
    class_counts = torch.zeros(2)
    nan_count = 0
    items_with_single_class = 0
    items_with_nans = 0

    for i in tqdm(range(len(dataset))):
        item = dataset[i]

        if "features" in item and item["features"] is not None:
            features = item["features"].squeeze()

            # Check for NaNs
            nan_mask = torch.isnan(features)
            item_nan_count = nan_mask.sum().item()
            nan_count += item_nan_count
            if item_nan_count > 0:
                items_with_nans += 1

            # Count classes
            valid_features = features[~nan_mask]
            if len(valid_features) > 0:
                unique_classes = torch.unique(valid_features)
                if len(unique_classes) == 1:
                    items_with_single_class += 1

                counts = torch.bincount(valid_features.long(), minlength=2)
                class_counts += counts
                total_points += len(valid_features)

    print(f"\nResults:")
    print(f"  Total points: {total_points:,}")
    print(f"  Total NaN values: {nan_count:,}")
    print(f"  Items with NaNs: {items_with_nans}/{len(dataset)}")
    print(f"  Items with single class: {items_with_single_class}/{len(dataset)}")
    print(
        f"  Class 0 (wood): {int(class_counts[0]):,} ({class_counts[0]/total_points*100:.1f}%)"
    )
    print(
        f"  Class 1 (leaf): {int(class_counts[1]):,} ({class_counts[1]/total_points*100:.1f}%)"
    )

    return class_counts, total_points


if __name__ == "__main__":
    pass

    # Analyze all three sites and all splits
    sites = [
        "data/tlspt_labelled/plot_octrees/hjfo-finl/hjfo-finl-splits.csv",
        "data/tlspt_labelled/plot_octrees/hjfo-poll/hjfo-poll-splits.csv",
        "data/tlspt_labelled/plot_octrees/hjfo-spal/hjfo-spal-splits.csv",
    ]

    for site_file in sites:
        print(f"\n{'='*60}")
        print(f"SITE: {site_file}")
        print(f"{'='*60}")

        for split in ["train", "val", "test"]:
            analyze_full_dataset(site_file, split, scale=2)
