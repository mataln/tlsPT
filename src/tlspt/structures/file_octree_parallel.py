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
        self,
        init_from: TLSPointclouds | list | PathOrStr,
        min_scale: int | None = None,
    ):  # Loads a file octree from a folder or initializes an empty one
        if isinstance(init_from, TLSPointclouds) or isinstance(init_from, list):
            logger.info("Building octree from pointclouds")
            if min_scale is None:
                raise ValueError(
                    "min_scale must be provided when initializing from pointclouds"
                )
            self.build_from_pointcloud_gpu(init_from, min_scale)
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

    def collect_leaf_nodes(self, root: _OctreeNode) -> list[tuple[torch.Tensor, str]]:
        """Collect all leaf nodes' bboxes and IDs from the octree"""
        leaf_data = []
        queue = deque([root])

        while queue:
            node = queue.pop()
            if node.is_leaf():
                leaf_data.append((node.bbox, node.id))
            else:
                queue.extendleft(node.children)

        return leaf_data

    def save_voxels(
        self,
        pointclouds: TLSPointclouds,
        out_folder: PathOrStr,
        insert_missing_features: bool = True,
        **kwargs,
    ):
        """
        GPU-accelerated partitioning with sequential saving
        """
        if isinstance(pointclouds, list) or (
            isinstance(pointclouds, TLSPointclouds)
            and len(pointclouds.points_list()) > 1
        ):
            logger.warning(
                "Building octree from multiple clouds. Check they are from the same scene"
            )
            pointclouds = join_pointclouds_as_scene(
                pointclouds, insert_missing_features=insert_missing_features
            )

        assert len(pointclouds.points_list()) == 1, "Multiple scenes detected"

        # Collect all leaf nodes
        logger.info("Collecting leaf nodes")
        leaf_nodes = self.collect_leaf_nodes(self.root)
        logger.info(f"Found {len(leaf_nodes)} voxels to process")

        # Move point data to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        points = pointclouds.points_packed().to(device)
        normals = (
            pointclouds.normals_packed().to(device)
            if pointclouds.normals_packed() is not None
            else None
        )
        features = (
            pointclouds.features_packed().to(device)
            if pointclouds.features_packed() is not None
            else None
        )

        # Ensure output directory exists
        os.makedirs(out_folder, exist_ok=True)

        # Create single reader instance
        reader = TR()
        num_saved = 0

        logger.info("Partitioning and saving voxels")
        for bbox, node_id in leaf_nodes:
            bbox_gpu = bbox.to(device)

            # Vectorized bounds check on GPU
            mins = bbox_gpu[:, 0].unsqueeze(0)
            maxs = bbox_gpu[:, 1].unsqueeze(0)
            in_bounds = ((points >= mins) & (points <= maxs)).all(dim=1)

            if torch.any(in_bounds):
                # Get points for this voxel
                voxel_points = points[in_bounds].cpu()
                voxel_normals = (
                    normals[in_bounds].cpu() if normals is not None else None
                )
                voxel_features = (
                    features[in_bounds].cpu() if features is not None else None
                )

                # Create TLSPointclouds object
                data = TLSPointclouds(
                    points=[voxel_points],
                    normals=[voxel_normals] if voxel_normals is not None else None,
                    features=[voxel_features] if voxel_features is not None else None,
                    feature_names=pointclouds._feature_names,
                )

                # Save to file
                out_path = os.path.join(out_folder, f"{node_id}.ply")
                reader.save_pointcloud(data, out_path, binary=True)

                num_saved += 1
                if num_saved % 100 == 0:
                    logger.info(f"Saved {num_saved} voxels")

        # Clean up GPU memory
        del points, normals, features
        torch.cuda.empty_cache()

        logger.info(f"Successfully saved {num_saved} voxels")

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

    def build_from_pointcloud_gpu(
        self, pointclouds: TLSPointclouds, min_scale: float = 1.0
    ):
        """
        Builds the octree using GPU acceleration for point operations
        """
        if isinstance(pointclouds, list) or (
            isinstance(pointclouds, TLSPointclouds)
            and len(pointclouds.points_list()) > 1
        ):
            logger.warning(
                "Building octree from multiple clouds. Check they are from the same scene"
            )
            pointclouds = join_pointclouds_as_scene(
                pointclouds, insert_missing_features=True
            )

        assert len(pointclouds.points_list()) == 1, "Multiple scenes detected"

        if len(pointclouds.points_list()[0]) == 0:
            raise ValueError("Passed empty pointclouds")

        self.feature_names = pointclouds._feature_names
        points = pointclouds.points_list()[0]

        # Move points to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points = points.to(device)

        # Initial bbox calculation
        bbox = pointclouds.get_bounding_boxes()[0]
        bbox_size = torch.max(bbox[:, 1] - bbox[:, 0])

        # Grow until slightly larger than next power of 2
        power = 1
        while power < bbox_size:
            power *= 2
        bbox_size = power + 0.01

        cubic_bbox = torch.zeros((3, 2))
        cubic_bbox[:, 0] = bbox[:, 0] - 0.05
        cubic_bbox[:, 1] = bbox[:, 0] + bbox_size + 0.05
        cubic_bbox = cubic_bbox.to(device)

        # Create root node
        self.root = self._OctreeNode(
            cubic_bbox.cpu()
        )  # Node stays on CPU but keeps GPU data reference
        self.root.num_points = len(points)

        def split_node_gpu(node_bbox: torch.Tensor, points: torch.Tensor) -> list[dict]:
            """Inner function to split a node using GPU operations"""
            mid = (node_bbox[:, 0] + node_bbox[:, 1]) / 2

            # Pre-calculate point positions relative to midpoint
            points_relative = points > mid.unsqueeze(0)  # shape: [N, 3]

            children_data = []
            # Process all potential children
            for i in range(8):
                bits = torch.tensor(
                    [(i >> j) & 1 for j in range(3)], dtype=torch.bool, device=device
                )

                # Points in this child are those that match the bit pattern
                matches = points_relative == bits.unsqueeze(0)
                in_bounds = matches.all(dim=1)
                points_in_child = points[in_bounds]

                if len(points_in_child) > 0:
                    # Calculate bbox for child
                    child_bbox_min = torch.where(bits, mid, node_bbox[:, 0])
                    child_bbox_max = torch.where(bits, node_bbox[:, 1], mid)
                    child_bbox = torch.stack([child_bbox_min, child_bbox_max], dim=1)

                    child_data = {
                        "bbox": child_bbox,
                        "points": points_in_child,
                        "num_points": len(points_in_child),
                    }
                    children_data.append(child_data)

            return children_data

        # Process levels
        current_level = [{"bbox": cubic_bbox, "points": points, "node": self.root}]

        while current_level:
            # Filter nodes that need splitting
            nodes_to_split = [
                node_data
                for node_data in current_level
                if node_data["node"].scale >= 2 * min_scale
            ]
            if not nodes_to_split:
                break

            next_level = []
            logger.info(f"Processing {len(nodes_to_split)} nodes")

            # Process each node in the current level
            for parent_data in nodes_to_split:
                children_data = split_node_gpu(
                    parent_data["bbox"], parent_data["points"]
                )

                if children_data:
                    parent_node = parent_data["node"]
                    parent_node.children = []

                    for child_data in children_data:
                        # Move bbox back to CPU for the node object
                        child_node = self._OctreeNode(child_data["bbox"].cpu())
                        child_node.num_points = child_data["num_points"]
                        parent_node.children.append(child_node)

                        if child_node.scale >= 2 * min_scale:
                            next_level.append(
                                {
                                    "bbox": child_data["bbox"],
                                    "points": child_data["points"],
                                    "node": child_node,
                                }
                            )

            current_level = next_level
            if current_level:
                logger.info(
                    f"Next level has {len(current_level)} nodes at scale {current_level[0]['node'].scale}"
                )

        # Clean up GPU memory
        torch.cuda.empty_cache()
