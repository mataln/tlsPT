from __future__ import annotations

import hashlib
import json
import os
import pathlib
from collections import deque

import h5py
import numpy as np
import torch
from loguru import logger
from pytorch3d.io.utils import PathOrStr

from tlspt.io.tls_reader import TLSReader as TR
from tlspt.structures.pointclouds import TLSPointclouds, join_pointclouds_as_scene


class FileOctree:
    """
    Class for octree based file storage access
    Folder structure:
    SPA01
    |---octree_SPA01.json #Might change name
    |---voxels/
        |---0fab44e.ply
        |---0fab44a.ply
        |---0fab44f.ply
        |---...
    """

    class _OctreeNode:
        def __init__(
            self,
            bbox: torch.tensor,
            children: list[FileOctree._OctreeNode] | None = None,
        ):
            self.bbox = bbox
            self.scale = torch.max(bbox[:, 1] - bbox[:, 0])
            self.children = children
            self.id = self.id_from_bbox()
            self.num_points = None

        def id_from_bbox(self):
            bbox_bytes = self.bbox.numpy().tobytes()
            hex_id = hashlib.md5(bbox_bytes).hexdigest()[:16]
            return hex_id

        def is_leaf(self):
            return self.children is None

        def count_points(self, pointclouds: TLSPointclouds):
            """
            Counts the number of points in the bounding box
            """
            if len(pointclouds.points_list()) != 1:
                raise ValueError("Multiple pointclouds detected during count_points")
            points_in_bbox = pointclouds.inside_box(self.bbox.T)
            self.num_points = torch.sum(points_in_bbox).item()
            return self.num_points

        def has_points(self, pointclouds: TLSPointclouds):
            """
            Checks if the pointclouds has any points in the bounding box
            """
            return self.count_points(pointclouds) > 0

        def split(self, pointclouds: TLSPointclouds | None = None):
            """
            Splits the node into 8 children
            """
            if self.is_leaf():
                # Create children nodes
                self.children = []
                for i in range(8):
                    child_bbox = self._get_child_bbox(i)
                    child_node = FileOctree._OctreeNode(child_bbox)
                    if pointclouds is not None and child_node.has_points(
                        pointclouds=pointclouds
                    ):  # checking if children have points
                        self.children.append(child_node)
                    elif pointclouds is None:  # not checking for points
                        self.children.append(child_node)

            else:
                raise ValueError("Node is not a leaf node")

        def _get_child_bbox(self, i: int):
            """
            Returns the bounding box of the ith child
            """
            mid = (self.bbox[:, 0] + self.bbox[:, 1]) / 2
            axes = torch.arange(3)

            bits = ((i >> axes) & 1).to(
                self.bbox.dtype
            )  # 3 bit number for 8 children. bit = left/right for each axis

            child_bbox_min = (
                self.bbox[:, 0] * (1 - bits) + mid * bits
            )  # bit = 0 -> min, bit = 1 -> mid
            child_bbox_max = (
                mid * (1 - bits) + self.bbox[:, 1] * bits
            )  # bit = 0 -> mid, bit = 1 -> max
            return torch.stack([child_bbox_min, child_bbox_max], dim=1)

        def to_dict(self):
            """
            Returns a dictionary representation of the node. Includes children recursively
            """
            return {
                "id": self.id,
                "bbox": self.bbox.tolist(),
                "scale": self.scale.item(),
                "num_points": self.num_points,
                "children": [child.to_dict() for child in self.children]
                if self.children is not None
                else None,
            }

        @classmethod
        def from_json(cls, json_file: PathOrStr):
            """
            Loads an octree from a json file
            """
            with open(json_file) as f:
                json_data = json.load(f)
                data = json_data["root"]
                feature_names = json_data["feature_names"]
            return cls.from_dict(data), feature_names

        @classmethod
        def from_dict(cls, data):
            bbox = torch.tensor(data["bbox"])
            node = cls(bbox)
            node.id = data["id"]
            node.num_points = data["num_points"]
            if data["children"]:
                node.children = [cls.from_dict(child) for child in data["children"]]
            else:
                node.children = None
            return node

    def __init__(
        self, init_from: TLSPointclouds | list | PathOrStr, min_scale: int | None = None
    ):  # Loads a file octree from a folder or initializes an empty one
        if isinstance(init_from, TLSPointclouds) or isinstance(init_from, list):
            logger.info("Building octree from pointclouds")
            if min_scale is None:
                raise ValueError(
                    "min_scale must be provided when initializing from pointclouds"
                )
            self.build_from_pointcloud(init_from, min_scale)
        else:
            logger.info(f"Initializing octree from {type(init_from)} {init_from}")
            if min_scale is not None:
                raise ValueError(
                    "min_scale must not be provided when initializing from folder"
                )
            if isinstance(init_from, (str, pathlib.Path)):
                self.root, self.feature_names = self._OctreeNode.from_json(init_from)
            else:
                raise ValueError(f"Invalid init_from type {type(init_from)}")
        return

    def save_voxels(
        self,
        pointclouds: TLSPointclouds | list[TLSPointclouds],
        out_folder: PathOrStr,
        insert_missing_features: bool = False,
        use_hdf5: bool = False,
        **kwargs,
    ):
        """
        Voxelizes pointclouds according to the octree and saves them to out_folder
        """
        if (
            isinstance(pointclouds, TLSPointclouds)
            and len(pointclouds.points_list()) > 1
        ) or (isinstance(pointclouds, list)):
            logger.warning(
                "Building octree from multiple clouds. Check they are from the same scene"
            )
            pointclouds = join_pointclouds_as_scene(
                pointclouds, insert_missing_features
            )

        assert len(pointclouds.points_list()) == 1, "Multiple scenes detected"

        logger.info(f"Saving voxels to {out_folder}. HDF5: {use_hdf5}")
        if use_hdf5:
            logger.warning("Saving using HDF5. HDF5 files will not contain normals.")

        if not use_hdf5:
            reader = TR()
        queue = deque()
        queue.appendleft(self.root)

        while queue:
            node = queue.pop()
            if node.is_leaf():
                points_in_voxel = pointclouds.inside_box(
                    node.bbox.T
                )  # Returns bool array over pointclouds
                if torch.sum(points_in_voxel) > 0:
                    if not use_hdf5:
                        data = TLSPointclouds(
                            points=[pointclouds.points_packed()[points_in_voxel]],
                            normals=[pointclouds.normals_packed()[points_in_voxel]]
                            if pointclouds.normals_packed() is not None
                            else None,
                            features=[pointclouds.features_packed()[points_in_voxel]]
                            if pointclouds.features_packed() is not None
                            else None,
                            feature_names=pointclouds._feature_names,
                        )
                        reader.save_pointcloud(
                            data,
                            os.path.join(out_folder, f"{node.id}.ply"),
                            binary=True,
                            **kwargs,
                        )
                    else:
                        data = {
                            "points": pointclouds.points_packed()[
                                points_in_voxel
                            ].numpy(),
                            "features": pointclouds.features_packed()[
                                points_in_voxel
                            ].numpy()
                            if pointclouds.features_packed() is not None
                            else None,
                        }

                        # Shuffle to avoid problems with chunked indexing
                        if data["points"].shape[0] != data["features"].shape[0]:
                            raise ValueError(
                                "Number of points and features do not match. Something has gone very wrong elsewhere."
                            )

                        shuffled_idx = np.random.permutation(data["points"].shape[0])
                        data["points"] = data["points"][shuffled_idx]
                        if data["features"] is not None:
                            data["features"] = data["features"][shuffled_idx]

                        self.save_hdf5(data, os.path.join(out_folder, f"{node.id}.h5"))
            else:
                queue.extendleft(node.children)

    def save(self, out_folder: PathOrStr, out_fname: str):
        """
        Saves the octree to a folder
        """
        if out_folder is None:
            raise ValueError("Output folder not set")

        out_file = os.path.join(out_folder, out_fname)

        with open(out_file, "w") as f:
            save_dict = {
                "feature_names": self.feature_names,
                "root": self.root.to_dict(),
            }
            json.dump(save_dict, f, indent=4)

        return

    def save_hdf5(self, data, out_file, dtype=np.float32, chunk_size=1024):
        """
        Saves the data to an hdf5 file
        """
        chunk_size = min(chunk_size, data["points"].shape[0])
        points_chunk_size = (chunk_size, 3)
        feature_chunk_size = (
            (chunk_size, data["features"].shape[1])
            if data["features"] is not None
            else None
        )

        with h5py.File(out_file, "w") as f:
            f.create_dataset(
                "points", data=data["points"].astype(dtype), chunks=points_chunk_size
            )

            if data["features"] is not None:
                f.create_dataset(
                    "features",
                    data=data["features"].astype(dtype),
                    chunks=feature_chunk_size,
                )

    def find_nodes_by_scale(self, scale: int):
        """
        Finds nodes with scale < node scale < 2*node scale
        """
        nodes = []
        queue = deque()
        queue.appendleft(self.root)
        while queue:
            node = queue.pop()
            if node.scale > scale and node.scale < 2 * scale:
                nodes.append(node)
            else:
                queue.extendleft(node.children)
        return nodes

    def find_leaves_under_node(self, node: _OctreeNode):
        """
        Finds all leaf nodes under a node
        """
        leaves = []
        queue = deque()
        queue.appendleft(node)
        while queue:
            node = queue.pop()
            if node.is_leaf():
                leaves.append(node)
            else:
                queue.extendleft(node.children)
        return leaves

    def build_from_pointcloud(
        self, pointclouds: TLSPointclouds | list[TLSPointclouds], min_scale: int = 1.0
    ):
        """
        Builds the octree from a TLSPointclouds object
        Assumes all the clouds are from the same scene
        Divides into 8 children successively until min_scale < bbox_size < 2*min_scale
        """
        if (
            isinstance(pointclouds, TLSPointclouds)
            and len(pointclouds.points_list()) > 1
        ) or (isinstance(pointclouds, list)):
            logger.warning(
                "Building octree from multiple clouds. Check they are from the same scene"
            )
            pointclouds = join_pointclouds_as_scene(
                pointclouds, insert_missing_features=True
            )  # Insert missing features must be on for joining. The features aren't actually used.

        assert len(pointclouds.points_list()) == 1, "Multiple scenes detected"

        if len(pointclouds.points_list()[0]) == 0:
            raise ValueError("Passed empty pointclouds")

        self.feature_names = pointclouds._feature_names

        # Get the bounding box
        bbox = pointclouds.get_bounding_boxes()[0]  # Shape 3,2 i.e [dimension, min/max]
        # Make square
        bbox_size = torch.max(bbox[:, 1] - bbox[:, 0])  # Maximum dimension
        logger.info(f"Initial bbox size: {bbox_size}")

        # Grow until slightly larger than next power of 2
        def next_power_of_2(bbox_size):
            """Returns the first power of 2 greater than bbox_size."""
            if bbox_size < 1:
                raise ValueError("bbox_size must be greater than 0.")
            power = 1
            while power < bbox_size:
                power *= 2
            return power

        bbox_size = next_power_of_2(bbox_size) + 0.01

        logger.info(f"Adjusted bbox size (before padding): {bbox_size}")

        # Make new cubic. Use padding
        cubic_bbox = torch.zeros((3, 2))
        cubic_bbox[:, 0] = bbox[:, 0] - 0.05
        cubic_bbox[:, 1] = bbox[:, 0] + bbox_size + 0.05

        # Create root node
        self.root = self._OctreeNode(cubic_bbox)
        self.root.count_points(pointclouds)
        current_scale = self.root.scale
        queue = deque()
        queue.appendleft(self.root)

        # Create tree breadth first until min_scale < bbox_size < 2*min_scale
        while queue:
            numat_scale = len(queue)
            for i in range(numat_scale):
                node = queue.pop()
                current_scale = node.scale

                if current_scale >= 2 * min_scale:
                    node.split(pointclouds)
                    queue.extendleft(node.children)  # Shouldn't add any empty children
                else:  # Reached scale, is leaf.
                    pass

        return
