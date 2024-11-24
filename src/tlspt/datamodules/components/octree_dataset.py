from __future__ import annotations

import multiprocessing as mp
import os

import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from tlspt.datamodules.components.base_site import BaseSiteDataset
from tlspt.io.tls_reader import TLSReader as TR
from tlspt.structures.file_octree import FileOctree
from tlspt.structures.pointclouds import TLSPointclouds, join_pointclouds_as_scene
from tlspt.utils import Hdf5Normalizer, TlsNormalizer, get_hash


class OctreeDataset(BaseSiteDataset):
    """
    Dataset for octree based file storage. One octree per plot.
    """

    def __init__(
        self,
        split_file: str,
        split: str,
        scale: int,
        feature_names: list = None,
        features_to_normalize: list = ["red", "green", "blue"],
        normalize: bool = True,
        transform=None,
        min_points: int = 512,
        plots_keep: list = None,
        in_memory: bool = False,
    ):
        """
        split_file: the csv file containing the split definitions
                    columns: identifier, split
                                09fas9f, train etc.
        split: one of 'train', 'test', 'val'
        """
        self.split_file = split_file
        self.split = split
        self.site_name = os.path.basename(self.split_file).split("-plot")[0]
        self.feature_names = feature_names
        self.normalize = normalize
        self.scale = scale
        self.transform = transform
        self.min_points = min_points
        self.plots_keep = plots_keep
        self.in_memory = in_memory

        super().__init__(split_file, split, self.site_name, "ply")

        # Load the octrees
        self.octree_files = self.find_octree_files()
        self.octrees = [FileOctree(f) for f in self.octree_files]

        # Work out which files need to be loaded
        # 1. Nodes at correct scale
        self.nodes = [
            octree.find_nodes_by_scale(self.scale) for octree in self.octrees
        ]  # Each entry is a list of nodes for a plot

        # 2. Filter by min_points
        self.nodes = [
            [node for node in plot_nodes if node.num_points >= self.min_points]
            for plot_nodes in self.nodes
        ]

        # 3. Find leaf nodes under each node
        self.leaf_nodes = [
            [
                self.octrees[i].find_leaves_under_node(node) for node in self.nodes[i]
            ]  # Size [num_plots][num nodes at scale][num leaves under node]
            for i in range(len(self.octrees))
        ]

        files_to_load = [
            [
                [
                    os.path.join(self.plot_folders[i], "voxels", f"{leaf_node.id}.ply")
                    for leaf_node in parent_node
                ]
                for parent_node in self.leaf_nodes[i]
            ]
            for i in range(len(self.octrees))
        ]  # Size [num_plots][num nodes at scale][num leaves under node]

        self.files_to_load = []
        for plot_files in files_to_load:
            self.files_to_load.extend(
                plot_files
            )  # Size [num_files][num leaves under node]

        # Init TLS reader
        self.reader = TR()

        # Init normalizer
        self.normalizer = TlsNormalizer(
            self,
            params={"features_to_normalize": features_to_normalize},
            n_samples=1000,
            out_dtype=torch.float32,
        )

        if self.in_memory:
            self.load_all_data()

    def __str__(self):
        return (
            f"{self.__class__.__name__}({self.split_file}, {self.split}, {self.scale})"
        )

    def __repr__(self):
        return get_hash(
            self.__class__.__name__
            + str(self.split_file)
            + str(self.split)
            + str(self.scale)
        )

    def prepare_data(self, force_compute=False):
        """
        Preprocessing step to be defined in each dataset.
        """
        if self.normalize:
            self.normalizer.prepare_data(force_compute=force_compute)
        return

    def find_octree_files(self):
        """
        Find the octree files in the folder
        """
        octree_files = []
        for folder in self.plot_folders:
            json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
            if len(json_files) != 1:
                raise ValueError(
                    f"Expected one json file in {folder}, found {len(json_files)}"
                )

            plot_name = os.path.split(json_files[0])[1].split(".")[0]
            if self.plots_keep is None or plot_name in self.plots_keep:
                octree_files.append(os.path.join(folder, json_files[0]))

        return octree_files

    @staticmethod
    def _load_item(args):
        """
        Wrapper for loading an item in parallel
        """
        dataset, idx = args
        return dataset.load_item(idx)

    def load_all_data(self, n_workers=24):
        """
        Preloads all data into memory
        """
        logger.info("Loading dataset into memory")
        args = [(self, idx) for idx in range(len(self.files_to_load))]

        if n_workers == -1:
            n_workers = mp.cpu_count() - 4
        chunksize = len(args) // n_workers
        logger.info(f"Using chunksize of {chunksize} files across {n_workers} workers")

        try:
            with mp.Pool(n_workers) as pool:
                # Create the iterator
                iterator = pool.imap(self._load_item, args, chunksize=chunksize)
                results = []

                # Process results one at a time to catch failures
                for result in tqdm(iterator, total=len(args)):
                    if result is None:
                        raise RuntimeError("Worker failed to load item")
                    results.append(result)

            self.data = results
            logger.info("Dataset loaded into memory")

        except Exception as e:
            logger.error(f"Error during parallel loading: {e}")
            # Make sure to terminate the pool
            pool.terminate()
            raise e

    def load_item(self, idx):
        """
        loads an item without transforms
        """
        voxel_leaf_nodes = self.files_to_load[idx]
        leaf_node_pointclouds = [
            self.reader.load_pointcloud(f) for f in voxel_leaf_nodes
        ]
        pc = join_pointclouds_as_scene(
            leaf_node_pointclouds, insert_missing_features=True
        )

        if self.feature_names is None:
            return TLSPointclouds(points=pc.points_packed().unsqueeze(0), features=None)
        else:
            # Create a feature array in the requested order, filling in missing features with NaNs
            feature_indices = [
                pc._feature_names.index(name) if name in pc._feature_names else -1
                for name in self.feature_names
            ]
            features = pc.features_packed()[:, feature_indices]
            if -1 in feature_indices:  # Fill in missing features with NaNs
                features[:, feature_indices.index(-1)] = np.nan
            return TLSPointclouds(
                points=pc.points_packed().unsqueeze(0),
                features=features.unsqueeze(0),
                feature_names=self.feature_names,
            )

    def __getitem__(self, idx):
        """
        loads an item from the dataset and normalizes it
        """
        if self.in_memory:
            pc = self.data[idx]
        else:
            pc = self.load_item(idx)

        if self.normalize:
            if self.normalizer.mean is None and self.feature_names is not None:
                raise ValueError("Normalizer not computed. Run prepare_data first.")
            pc = self.normalizer.normalize(pc)

        datapoint = (
            {"points": pc.points_packed(), "features": pc.features_packed()}
            if pc.features_packed() is not None
            else {"points": pc.points_packed()}
        )

        datapoint["lengths"] = datapoint["points"].shape[0]

        if self.transform:
            datapoint = self.transform(datapoint)

        return datapoint

    def __len__(self):
        return len(self.files_to_load)


class OctreeDatasetHdf5(OctreeDataset):
    """
    Dataset for octree based file storage. One octree per plot.
    """

    def __init__(
        self,
        split_file: str,
        split: str,
        scale: int,
        idx_sampler,
        feature_names: list = None,
        features_to_normalize: list = ["red", "green", "blue"],
        normalize: bool = True,
        transform=None,
        min_points: int = 512,
        plots_keep: list = None,
    ):
        """
        split_file: the csv file containing the split definitions
                    columns: identifier, split
                                09fas9f, train etc.
        split: one of 'train', 'test', 'val'
        """
        super().__init__(
            split_file,
            split,
            scale,
            feature_names,
            features_to_normalize,
            normalize,
            transform,
            min_points,
            plots_keep,
        )
        self.idx_sampler = idx_sampler

        # self.leaf_nodes [num_plots][num nodes at scale][num leaves under node]
        self.leaf_nodes_flattened = []
        for plot_nodes in self.leaf_nodes:
            self.leaf_nodes_flattened.extend(
                plot_nodes
            )  # Size [num_plots*num nodes at scale][num leaves under node]

        files_to_load = [
            [
                [
                    os.path.join(
                        self.plot_folders[i], "voxels_hdf5", f"{leaf_node.id}.h5"
                    )
                    for leaf_node in parent_node
                ]
                for parent_node in self.leaf_nodes[i]
            ]
            for i in range(len(self.octrees))
        ]

        self.files_to_load = []
        for plot_files in files_to_load:
            self.files_to_load.extend(plot_files)

        # Init normalizer
        self.normalizer = Hdf5Normalizer(
            self,
            params={"features_to_normalize": features_to_normalize},
            n_samples=1000,
            out_dtype=torch.float32,
        )

    def load_item(self, idx):
        """
        loads an item without transforms
        """
        # List of h5 files
        voxel_leaf_nodes = self.files_to_load[idx]
        if len(voxel_leaf_nodes) != 1:
            raise NotImplementedError(
                f"Multi-voxel loading is not supported for Hdf5 files. Got {len(voxel_leaf_nodes)} files."
            )
        h5_file = voxel_leaf_nodes[0]

        # Get the length of each file
        # length = self.leaf_nodes_flattened[idx].num_points

        # Load the h5 file
        with h5py.File(h5_file, "r") as f:
            points_dset = f["points"]
            chunk_size = points_dset.chunks[0]
            length = points_dset.shape[0]

            # Sample
            slices = self.idx_sampler(chunk_size, length)
            points = []
            for start_idx, end_idx in slices:
                data_slice = points_dset[start_idx:end_idx]
                points.append(data_slice)

            features = None
            if self.feature_names is not None:
                features_dset = f["features"]
                features = []
                for start_idx, end_idx in slices:
                    data_slice = features_dset[start_idx:end_idx]
                    features.append(data_slice)

            points = torch.as_tensor(np.concatenate(points, axis=0))
            if features is not None:
                features = torch.as_tensor(np.concatenate(features, axis=0))

        return points, features

    def __getitem__(self, idx):
        """
        loads an item from the dataset and normalizes it
        """
        points, features = self.load_item(idx)
        scale = self.leaf_nodes_flattened[idx][0].scale

        if self.feature_names is not None:  # Select features
            feature_indices = [
                self.feature_names.index(name) for name in self.feature_names
            ]
            features = features[:, feature_indices]

        datapoint = (
            {"points": points, "features": features, "lengths": points.shape[0]}
            if features is not None
            else {"points": points, "lengths": points.shape[0]}
        )

        if self.normalize:
            if self.normalizer.mean is None and self.feature_names is not None:
                raise ValueError("Normalizer not computed. Run prepare_data first.")
            datapoint = self.normalizer.normalize(datapoint, scale)

        if self.transform:
            datapoint = self.transform(datapoint)

        return datapoint

    def __len__(self):
        return len(self.files_to_load)
