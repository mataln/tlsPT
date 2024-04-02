from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from tlspt import utils


class BaseDataset(Dataset):

    """
    Base class for all datasets
    """

    def __init__(
        self,
        split_file: str,
        split: str,
        dataset: str,
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

        if not utils.check_file_exists(split_file):
            raise ValueError(f"cannot find split file at {split_file}")

        self.split_file = split_file
        self.dataset_folder = f'{split_file.split("_splits")[0]}/{dataset}'
        self.split = split
        self.dataset = dataset

        if not utils.check_dir_exists(self.dataset_folder):
            raise ValueError(f"cannot find dataset at {self.dataset_folder}")

        logger.info(f"{self}: reading splits from {split_file}")
        self.splits = pd.read_csv(split_file, index_col="identifier")
        self.splits = self.splits[self.splits.split == self.split].copy()

        if len(self.splits) == 0:
            raise ValueError(f"{self}: no splits for '{split}' found in {split_file}")

        # makes file list with files actually found in folder
        expected_files = [f"{i}.{self.tile_format}" for i in self.splits.index]
        logger.info(
            f"{self}: looking for {len(expected_files)} files in {self.dataset_folder}"
        )
        self.files = utils.list_all_files(self.dataset_folder)
        self.files = [f.split("/")[-1] for f in self.files]
        self.files = sorted(list(set(self.files).intersection(expected_files)))

        self.check_discrepancies_between_expected_files_and_folder_files(expected_files)

        self.tileids = [i.split(".")[0] for i in self.files]

        self.files = np.array(self.files, dtype=np.string_)
        self.tileids = np.array(self.tileids, dtype=np.string_)

        if len(self.files) > 0:
            logger.info(
                f"{self}: found {len(self.files)} tiles for '{self.split}' out of {len(expected_files)} tiles defined in split file"
            )
        else:
            raise ValueError(
                f"{self}: no '{self.split}' files found in {self.dataset_folder}"
            )

    def prepare_data(self):
        """This is an optional preprocessing step to be defined in each dataloader.
        It will be called by the Pytorch Lighting Data Module.
        All the preprocessing steps such as normalization or transformations
        might be write in this method when you override it.
        """

    def __len__(self):
        return len(self.files)

    def get_files(self):
        """
        returns the decoded set of file names delivered by this dataset
        """
        return [i.decode("utf-8") for i in self.files]

    def get_tileids(self):
        """
        returns the decoded set of ids of the tiles delivered by this dataset
        """
        return [
            i.decode("utf-8") if isinstance(i, np.bytes_) else i for i in self.tileids
        ]

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
        number_of_discrepancies = len(set(expected_files).difference(self.files))
        if number_of_discrepancies > 0:
            logger.warning(f"{self}: There are {number_of_discrepancies} discrepancies")
