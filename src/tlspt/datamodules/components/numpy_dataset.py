from __future__ import annotations

from tlspt import io

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
        voxel_format: str = "npy",
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
        if num_channels < 4:
            raise ValueError("num_channels must be at least 4")

        if voxel_format not in ["npy"]:
            raise ValueError(f"unsupported tile format {voxel_format}")

        super().__init__(split_file, split)

        self.transform = transform

        # Last step - init normalizer

    def prepare_data(self):
        if self.normalize:
            self.normalizer.prepare_data()

    def load_item(self, idx):
        """
        loads an item without transforms
        """

        file_path = f"{self.dataset_folder}/{self.get_files()[idx]}"

        arr = io.load_numpy(file_path)

        if arr.shape[0] != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} channels, got {arr.shape[0]}"
            )

        return arr

    def __getitem__(self, idx):
        """
        loads an item from the dataset and normalizes it
        """
        arr = self.load_item(idx)

        if self.normalize:
            arr = self.normalizer.normalize(arr)

        if self.transform is not None:
            arr = self.transform(arr)

        return arr

    def __len__(self):
        return len(self.files)
