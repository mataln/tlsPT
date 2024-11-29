# Script to convert a plot point cloud to voxels

from __future__ import annotations

import argparse
import os

import numpy as np
from loguru import logger

# from pytorch3d.io import IO
from tlspt.io.tls_reader import TLSReader as TR
from tlspt.structures.pointclouds import join_pointclouds_as_scene


def subdivide_bbox(bbox, scale):
    """
    Vectorized subdivision of a bounding box into smaller boxes of given scale.

    Args:
        bbox: Array of shape (3,2) where first dim is xyz, second is min/max
        scale: Size of smaller boxes (same for all dimensions)

    Returns:
        mins: Array of shape (N,3) containing minimum corners of all sub-boxes
        scale: Float, size of boxes (uniform in all dimensions)
    """
    # Calculate number of boxes needed in each dimension
    dims = bbox[:, 1] - bbox[:, 0]
    n_boxes = np.ceil(dims / scale).astype(int)

    # Create grid of minimum corners
    coords = [np.arange(n) * scale + bbox[i, 0] for i, n in enumerate(n_boxes)]
    mins = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1).reshape(-1, 3)

    return mins, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--octree-fname", type=str, default="meta_naive.json")
    parser.add_argument("--min_scale", type=float, default=2.0)
    parser.add_argument("--hdf5", action="store_true")
    parser.add_argument("--v", action="store_true")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    args.out_fname
    args.min_scale

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "voxels"), exist_ok=True)

    logger.info(f"Searching for .ply files in {input_folder}")
    ply_files = [x for x in os.listdir(input_folder) if x.endswith(".ply")]
    assert (
        len(ply_files) == 1
    ), "Only one .ply file should be in the input folder. Use the other script for more"
    logger.info(f"Found {len(ply_files)} .ply files")

    reader = TR()

    def read(file):
        return reader.load_pointcloud(
            path=os.path.join(input_folder, file), device="cpu"
        )

    logger.info("Reading point clouds")
    pointclouds = []
    failed = 0
    for file in ply_files:
        try:
            pointclouds.append(read(file))
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            failed += 1
    logger.info(f"Failed to read {failed} point clouds")
    # pointclouds = [read(file) for file in ply_files]

    main_pc = join_pointclouds_as_scene(pointclouds)
    main_bbox = main_pc.get_bounding_boxes()[0]  # Shape 3,2 i.e [dimension, min/max]
    logger.info(f"bbox size {main_bbox.size()}")

    # Generate 2m voxels covering the bounding box - TODO
    logger.info("Generating voxels")

    # Save each voxel - TODO

    logger.info("Saving voxels")
    out_folder = (
        os.path.join(output_folder, "voxels_hdf5")
        if args.hdf5
        else os.path.join(output_folder, "voxels")
    )

    os.makedirs(out_folder, exist_ok=True)

    # Save here
    logger.info(f"voxels saved to {output_folder}")

    # Save to test.ply
    # reader.save_pointcloud(
    #     pc,
    #     os.path.join(output_folder, "test.ply"),
    #     remove=["red", "green", "blue", "scalar_distance"],
    #     colors_as_uint8=True,
    # )

    # # Load and print to check
    # pc2 = reader.load_pointcloud(
    #     path=os.path.join(output_folder, "test.ply"), device="cpu"
    # )


if __name__ == "__main__":
    main()
