"""
Scripts to generate splits.csv for octrees

File structure:

|---data/
    |---plot_octrees/
        |---site1/ [site folder]
            |---voxel_splits-train-val-test.csv #To generate. train=0.6, val=0.2, test=0.2
            |---octrees/
                |---plot1/
                    |---plot1.json
                    |---voxels/
                        |---0f0a.ply
                        |---....
                |---plot2
                    |---...
        |---site2/
            |---...

"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)  # Input SITE folder
    parser.add_argument(
        "--ratios", nargs=3, default=[0.7, 0.15, 0.15], required=True
    )  # Train, Val, Test
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    input_folder = args.input_folder
    ratios = [float(x) for x in args.ratios]
    seed = args.seed

    # Site is directory name two level above input_folder
    site = os.path.basename(os.path.dirname(os.path.dirname(input_folder)))

    if sum(ratios) != 1 or len(ratios) != 3:
        raise ValueError("Sum of ratios must be 1")

    # Get all plot folders
    plot_folders = [
        x
        for x in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, x))
    ]
    logger.info(f"Found {len(plot_folders)} plot folders")

    splits = pd.DataFrame(columns=["plot", "split"])

    # Add plot_folders to splits under plot
    for plot in plot_folders:
        splits = pd.concat([splits, pd.DataFrame({"plot": [plot]})])

    # Reassign splits to train, val, test
    # Shuffle splits
    splits = splits.sample(frac=1, random_state=seed)

    # Force-stratifed splits
    # Get number of plots
    num_plots = len(splits)
    num_train = int(num_plots * ratios[0])
    num_val = int(num_plots * ratios[1])
    num_test = num_plots - num_train - num_val

    logger.info(f"Site: {site}")
    logger.info(f"Total plots: {num_plots}")
    logger.info(f"Train: {num_train}, Val: {num_val}, Test: {num_test}")

    splits["split"] = np.concatenate(
        [
            np.repeat("train", num_train),
            np.repeat("val", num_val),
            np.repeat("test", num_test),
        ]
    )
    logger.info(f"Split distribution: \n{splits['split'].value_counts()}")

    splits = splits.sample(frac=1, random_state=seed)
    splits = splits.reset_index(drop=True)

    # Save splits to csv
    splits.to_csv(
        os.path.join(
            input_folder,
            f"{site}-plot_splits-tr{ratios[0]}-val{ratios[1]}-te{ratios[2]}_seed{seed}.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
