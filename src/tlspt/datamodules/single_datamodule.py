from __future__ import annotations

import torch
from hydra.utils import get_class
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, SubsetRandomSampler


class SingleDataModule(
    LightningDataModule
):  # Lightning data module to wrap one input dataset
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        limit_train_pct: float = 1.0,
        limit_val_pct: float = 1.0,
        limit_test_pct: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        train_class = get_class(train_dataset.target_class)
        self.train_dataset = train_class(**train_dataset.kwargs)

        val_class = get_class(val_dataset.target_class)
        self.val_dataset = val_class(**val_dataset.kwargs)

        test_class = get_class(test_dataset.target_class)
        self.test_dataset = test_class(**test_dataset.kwargs)

        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None

        if 0 < limit_train_pct < 1.0:
            num_samples = int(len(self.train_dataset) * limit_train_pct)
            train_indices = torch.randperm(len(self.train_dataset))[:num_samples]
            self.train_sampler = SubsetRandomSampler(train_indices)
        elif limit_train_pct != 1.0:
            raise ValueError("limit_train_pct must be > 0 and <= 1")

        if 0 < limit_val_pct < 1.0:
            num_samples = int(len(self.val_dataset) * limit_val_pct)
            val_indices = torch.randperm(len(self.val_dataset))[:num_samples]
            self.val_sampler = SubsetRandomSampler(val_indices)
        elif limit_val_pct != 1.0:
            raise ValueError("limit_val_pct must be > 0 and <= 1")

        if 0 < limit_test_pct < 1.0:
            num_samples = int(len(self.test_dataset) * limit_test_pct)
            test_indices = torch.randperm(len(self.test_dataset))[:num_samples]
            self.test_sampler = SubsetRandomSampler(test_indices)
        elif limit_test_pct != 1.0:
            raise ValueError("limit_test_pct must be > 0 and <= 1")

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
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=self.val_sampler,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=self.test_sampler,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )
