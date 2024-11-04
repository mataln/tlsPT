# Script to convert a folder of individual tree point clouds (.ply) to an octree for that plot

from __future__ import annotations

import argparse
import os

from loguru import logger

# from pytorch3d.io import IO
from tlspt.io.tls_reader import TLSReader as TR
from tlspt.structures.file_octree import FileOctree as FOctree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--octree-fname", type=str, default="octree.json")
    parser.add_argument("--min_scale", type=float, default=1.5)
    parser.add_argument("--v", type=bool, default=False)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    octree_fname = args.octree_fname
    min_scale = args.min_scale

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "voxels"), exist_ok=True)

    ply_files = [x for x in os.listdir(input_folder) if x.endswith(".ply")]

    reader = TR()

    def read(file):
        return reader.load_pointcloud(
            path=os.path.join(input_folder, file), device="cpu"
        )

    pointclouds = [read(file) for file in ply_files]

    octree = FOctree(
        pointclouds, min_scale=min_scale
    )  # Scale of leaves will be in min_scale to 2*min_scale
    octree.save(out_folder=output_folder, out_fname=octree_fname)
    octree.save_voxels(
        pointclouds,
        out_folder=os.path.join(output_folder, "voxels"),
        insert_missing_features=True,
    )

    logger.info("Octree saved to {output_folder}")

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
