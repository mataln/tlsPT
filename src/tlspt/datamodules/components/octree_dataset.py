from __future__ import annotations

import os

from tlspt.datamodules.components.base_site import BaseSiteDataset
from tlspt.io.tls_reader import TLSReader as TR
from tlspt.structures.file_octree import FileOctree
from tlspt.structures.pointclouds import join_pointclouds_as_scene


class OctreeDataset(BaseSiteDataset):
    """
    Base class for single-site datasets
    """

    def __init__(
        self,
        split_file: str,
        split: str,
        scale: int,
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

        super().__init__(split_file, split, self.site_name, "ply")

        # Load the octrees
        self.octree_files = self.find_octree_files()
        self.octrees = [FileOctree(f) for f in self.octree_files]

        # Work out which files need to be loaded
        self.nodes = [
            octree.find_nodes_by_scale(scale) for octree in self.octrees
        ]  # Each entry is a list of nodes for a plot

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
        ]

        self.files_to_load = []
        for plot_files in files_to_load:
            self.files_to_load.extend(plot_files)

        self.reader = TR()

    def prepare_data(self):
        """
        Preprocessing step to be defined in each dataset.
        """
        raise NotImplementedError("Not implemented for base site dataset")
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

            octree_files.append(os.path.join(folder, json_files[0]))

        return octree_files

    def load_item(self, idx):
        """
        loads an item without transforms
        """
        voxel_leaf_nodes = self.files_to_load[idx]
        leaf_node_pointclouds = [
            self.reader.load_pointcloud(f) for f in voxel_leaf_nodes
        ]
        return join_pointclouds_as_scene(
            leaf_node_pointclouds, insert_missing_features=True
        )

    def __getitem__(self, idx):
        """
        loads an item from the dataset and normalizes it
        """
        pc = self.load_item(idx)

        # if self.normalizer.mean is None:
        #     raise ValueError("Normalizer not computed. Run prepare_data first.")
        # if self.normalize:
        #     arr = self.normalizer.normalize(arr)

        # if self.transform is not None:
        #     arr = self.transform(arr)

        return pc

    def __len__(self):
        return len(self.files_to_load)
