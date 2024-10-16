from __future__ import annotations

from hydra.utils import get_class
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader


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
    ):
        super().__init__()

        train_class = get_class(train_dataset.target_class)
        self.train_dataset = train_class(**train_dataset.kwargs)

        val_class = get_class(val_dataset.target_class)
        self.val_dataset = val_class(**val_dataset.kwargs)

        test_class = get_class(test_dataset.target_class)
        self.test_dataset = test_class(**test_dataset.kwargs)

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
            shuffle=True,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
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
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )
