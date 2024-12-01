from __future__ import annotations

import os

import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from tlspt import utils


class BaseSiteDataset(Dataset):

    """
    Base class for single-site datasets
    """

    def __init__(
        self,
        split_file: str,
        split: str,
        dataset: str,
        voxel_format: str,
    ):
        """
        split_file: the csv file containing the split definitions
                    columns: identifier, split
                                09fas9f, train etc.
        split: one of 'train', 'test', 'val'
        dataset: name of the dataset
        """
        if not split in ["test", "train", "val"]:
            raise ValueError(
                f"invalid split '{split}', must be one of 'train', 'test' or 'val'"
            )

        if voxel_format not in ["npy", "ply"]:
            raise ValueError(f"unsupported voxel format {voxel_format}")

        if not utils.check_file_exists(split_file):
            raise ValueError(f"cannot find split file at {split_file}")

        self.voxel_format = voxel_format
        self.split_file = split_file
        self.base_folder = f"{os.path.split(self.split_file)[0]}/"
        self.split = split
        self.dataset = dataset

        logger.info(f"{self}: reading splits from {split_file}")
        plots = pd.read_csv(split_file, dtype={"plot": str, "split": str})
        plots = plots[plots.split == self.split].copy()["plot"].values

        if len(plots) == 0:
            logger.warning(
                f"{self}: no plots for '{split}' found for {split_file}. Dataset will be empty"
            )
            self.plots = []
            self.plot_folders = []
            return

        logger.info(f"{self}: looking for {len(plots)} folders in {self.base_folder}")
        self.plots = utils.list_all_folders(self.base_folder)
        self.plots = [f.split("/")[-1] for f in self.plots]
        self.plots = sorted(list(set(self.plots).intersection(plots)))

        self.check_discrepancies_between_expected_files_and_folder_files(plots)

        if len(self.plots) > 0:
            logger.info(
                f"{self}: found {len(self.plots)} plots for '{self.split}' out of {len(plots)} plots defined in split file"
            )
        else:
            raise ValueError(
                f"{self}: no '{self.split}' files found in {self.base_folder} due to discrepancies between split file and folder files"
            )

        self.plot_folders = self.get_plot_folders()

    def prepare_data(self):
        """
        Preprocessing step to be defined in each dataset.
        """
        raise NotImplementedError("Not implemented for base site dataset")
        return

    def __len__(self):
        raise NotImplementedError("Not implemented for base site dataset")
        return

    def get_plot_folders(self):
        """
        returns the folder for each plot
        """
        return [os.path.join(self.base_folder, plot) for plot in self.plots]

    def get_voxelids(self):
        raise NotImplementedError("Not implemented for base site dataset")

    def __repr__(self):
        return self.dataset

    def check_discrepancies_between_expected_files_and_folder_files(
        self, expected_files
    ):
        """
        Show to the user the discrepancies between expected_files from the
        splits and the real folder files

        expected_files: name list of files in the directory
        """
        number_of_discrepancies = len(set(expected_files).difference(self.plots))
        if number_of_discrepancies > 0:
            logger.warning(f"{self}: There are {number_of_discrepancies} discrepancies")
