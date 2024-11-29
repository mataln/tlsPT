from __future__ import annotations

import json
import os

import numpy as np
import torch

from tlspt.datamodules.components.base_site import BaseSiteDataset
from tlspt.io.tls_reader import TLSReader as TR
from tlspt.structures.pointclouds import TLSPointclouds, join_pointclouds_as_scene
from tlspt.utils import TlsNormalizer, get_hash


class NaiveDataset(BaseSiteDataset):
    """
    Dataset for naive voxels.
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

        super().__init__(split_file, split, self.site_name, "ply")

        # Load the metadata
        self.meta_files = self.find_metadata()

        def json_to_dict(json_file):
            with open(json_file) as f:
                return json.load(f)

        self.metadata = [json_to_dict(f) for f in self.meta_files]

        # dict['feature_names']
        # dict['voxels']['id'][id/bbox/scale/num_points]

        # 2. Filter by min_points
        self.nodes = [
            [node for node in meta["voxels"] if node.num_points >= self.min_points]
            for meta in self.metadata
        ]  # Size [num_plots][num_nodes_in_plot]

        files_to_load = [
            [
                os.path.join(self.plot_folders[i], "voxels", f"{node.id}.ply")
                for node in self.nodes[i]
            ]
            for i in range(len(self.nodes))
        ]

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

    def find_metadata(self):
        """
        Find the metadata files in the folder
        """
        json_files = []
        for folder in self.plot_folders:
            json_files = [f for f in os.listdir(folder) if f.endswith("_naive.json")]
            if len(json_files) != 1:
                raise ValueError(
                    f"Expected one json file in {folder}, found {len(json_files)}"
                )

            plot_name = os.path.split(json_files[0])[1].split(".")[0]
            if self.plots_keep is None or plot_name in self.plots_keep:
                json_files.append(os.path.join(folder, json_files[0]))

        return json_files

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
        pc = self.load_item(idx)

        if self.normalize:
            if self.normalizer.mean is None and self.feature_names is not None:
                raise ValueError("Normalizer not computed. Run prepare_data first.")
            pc = self.normalizer.normalize(pc, self.scale)

        datapoint = (
            {"points": pc.points_packed(), "features": pc.features_packed()}
            if pc.features_packed() is not None
            else {"points": pc.points_packed()}
        )

        datapoint["lengths"] = datapoint["points"].shape[0]  # Before transform

        if self.transform:
            datapoint = self.transform(
                datapoint
            )  # Transform should operate on points and features and lengths

        return datapoint

    def __len__(self):
        return len(self.files_to_load)
