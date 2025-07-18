from __future__ import annotations

from tlspt.io import io
from tlspt.utils import TlsNormalizer

from . import base


class NumpyDataset(base.BaseDataset):
    """
    Dataset for segmentation tasks
    """

    def __init__(
        self,
        split_file: str,
        split: str,
        num_channels: int = 4,  # Note - includes labels so 4 for 3d data w/o reflectance etc.
        has_labels: bool = False,
        normalize: bool = True,
        transform=None,
    ):
        """
        split_file: the csv file containing the split definitions
                    columns: identifier, split
                                09fas9f, train etc.
        split: one of 'train', 'test', 'val'
        dataset: name of the dataset
        """
        if num_channels < 3 + int(has_labels):
            raise ValueError(
                f"num_channels must be at least {3 + int(has_labels)} with labels: {has_labels}"
            )

        self.num_channels = num_channels
        self.dataset = f"numpy_{num_channels}ch"

        super().__init__(split_file, split, self.dataset, voxel_format="npy")

        self.normalize = normalize
        self.transform = transform

        # Last step - init normalizer
        self.normalizer = TlsNormalizer(
            self,
            params={"num_channels": num_channels, "has_labels": has_labels},
        )

    def prepare_data(self, force_compute=False):
        if self.normalize:
            self.normalizer.prepare_data(force_compute=force_compute)

    def load_item(self, idx):
        """
        loads an item without transforms
        """

        file_path = f"{self.dataset_folder}/{self.get_files()[idx]}"

        arr = io.load_numpy(file_path)

        if arr.ndim != 2:
            raise ValueError(f"expected 2D array, got shape {arr.shape}")

        if arr.shape[1] != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} channels, got {arr.shape[1]}"
            )

        return arr

    def __getitem__(self, idx):
        """
        loads an item from the dataset and normalizes it
        """
        arr = self.load_item(idx)

        if self.normalizer.mean is None:
            raise ValueError("Normalizer not computed. Run prepare_data first.")
        if self.normalize:
            arr = self.normalizer.normalize(arr)

        if self.transform is not None:
            arr = self.transform(arr)

        return arr

    def __len__(self):
        return len(self.files)
