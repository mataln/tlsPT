# Script to convert a folder of individual tree point clouds (.ply) to an octree for that plot

from __future__ import annotations

import argparse
import os

# from pytorch3d.io import IO
from tlspt.io.tls_reader import TLSReader as TR
from tlspt.structures.file_octree import FileOctree as FOctree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--v", type=bool, default=False)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    ply_files = [x for x in os.listdir(input_folder) if x.endswith(".ply")]

    reader = TR()

    def read(file):
        return reader.load_pointcloud(
            path=os.path.join(input_folder, file), device="cpu"
        )

    pointclouds = [read(file) for file in ply_files[:10]]

    octree = FOctree()
    octree.build_from_pointcloud(pointclouds)

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
