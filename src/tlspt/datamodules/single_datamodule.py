from __future__ import annotations

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader


class SingleDataModule(
    LightningDataModule
):  # Lightning data module to wrap one input dataset
    def __init__(
        self,
        dataset,
        dataset_kwargs,
        split_file,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ):
        super().__init__()

        self.train_dataset = dataset(
            split_file=split_file, split="train", **dataset_kwargs
        )

        self.val_dataset = dataset(split_file=split_file, split="val", **dataset_kwargs)

        self.test_dataset = dataset(
            split_file=split_file, split="test", **dataset_kwargs
        )

    def prepare_data(self):
        self.train_dataset.prepare_data()  # Precomputes mean + std for normalizers
        self.val_dataset.prepare_data()
        self.test_dataset.prepare_data()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle_train,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle_train,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle_train,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )
