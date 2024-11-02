from __future__ import annotations

import hashlib
from collections import deque

import torch
from loguru import logger
from pytorch3d.io.utils import PathOrStr
from pytorch3d.structures import join_pointclouds_as_scene

from tlspt.structures.pointclouds import TLSPointclouds


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

        def id_from_bbox(self):
            bbox_bytes = self.bbox.numpy().tobytes()
            hex_id = hashlib.md5(bbox_bytes).hexdigest()[:16]
            return hex_id

        def is_leaf(self):
            return self.children is None

        def has_points(self, pointclouds: TLSPointclouds):
            """
            Checks if the pointclouds has any points in the bounding box
            """
            if len(pointclouds.points_list()) != 1:
                raise ValueError("Multiple pointclouds detected during check_empty")
            points_in_bbox = pointclouds.inside_box(self.bbox.T)
            return any(points_in_bbox)

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

    def __init__(
        self, out_folder: PathOrStr | None = None
    ):  # Loads a file octree from a folder or initializes an empty one
        self.out_folder = out_folder

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
            and len(pointclouds.points_list()) == 1
        ) or (isinstance(pointclouds, list)):
            logger.warning(
                "Building octree from multiple clouds. Check they are from the same scene"
            )
            pointclouds = join_pointclouds_as_scene(pointclouds)

        assert len(pointclouds.points_list()) == 1, "Multiple scenes detected"

        # Get the bounding box
        bbox = pointclouds.get_bounding_boxes()[0]  # Shape 3,2 i.e [dimension, min/max]
        # Make square
        bbox_size = torch.max(bbox[:, 1] - bbox[:, 0])
        # Make new cubic bbox with 5cm padding
        cubic_bbox = torch.zeros((3, 2))
        cubic_bbox[:, 0] = bbox[:, 0] - 0.05
        cubic_bbox[:, 1] = bbox[:, 0] + bbox_size + 0.05

        # Create root node
        root = self._OctreeNode(cubic_bbox)
        current_scale = root.scale
        queue = deque()
        queue.appendleft(root)

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
