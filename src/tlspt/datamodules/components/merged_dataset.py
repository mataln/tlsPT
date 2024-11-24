from __future__ import annotations

from torch.utils.data import Dataset

from tlspt.datamodules.components.octree_dataset import OctreeDataset, OctreeDatasetHdf5
from tlspt.utils import get_hash


class MergedOctreeDataset(Dataset):
    def __init__(
        self,
        split_files: list,
        split: str,
        scales: list,
        feature_names: list = None,
        features_to_normalize: list = ["red", "green", "blue"],
        normalize: bool = True,
        min_points: list = None,  # Defaults to 512 each later
        transform=None,
        hdf5: bool = False,
        idx_sampler=None,
        in_memory: bool = False,
    ):
        if len(split_files) != len(scales):
            raise ValueError("Split files and scales must have the same length")

        if hdf5 and idx_sampler is None:
            raise ValueError("HDF5 requires an HDF5ChunkSampler")

        if hdf5 and in_memory:
            raise NotImplementedError("Preloading not supported for HDF5 datasets")

        if min_points is None:
            min_points = [512] * len(split_files)

        if hdf5:
            self.datasets = [
                OctreeDatasetHdf5(
                    split_file=split_files[i],
                    split=split,
                    scale=scales[i],
                    feature_names=feature_names,
                    features_to_normalize=features_to_normalize,
                    normalize=normalize,
                    transform=transform,
                    idx_sampler=idx_sampler,
                    min_points=min_points[i],
                )
                for i in range(len(split_files))
            ]
        else:
            self.datasets = [
                OctreeDataset(
                    split_file=split_files[i],
                    split=split,
                    scale=scales[i],
                    feature_names=feature_names,
                    features_to_normalize=features_to_normalize,
                    normalize=normalize,
                    transform=transform,
                    min_points=min_points[i],
                    in_memory=in_memory,
                )
                for i in range(len(split_files))
            ]

        self.idxs = []
        for i, d in enumerate(self.datasets):
            self.idxs.extend([(i, idx) for idx in range(len(d))])

    def prepare_data(self):
        for d in self.datasets:
            d.prepare_data()

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __repr__(self):
        return get_hash(
            f"{self.__class__.__name__}({[x.__repr__() for x in self.datasets]})"
        )

    def __getitem__(self, idx):
        dataset_idx, idx = self.idxs[idx]
        return self.datasets[dataset_idx][idx]
