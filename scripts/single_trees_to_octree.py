# Script to convert a folder of individual tree point clouds (.ply) to an octree for that plot

from __future__ import annotations

import argparse
import os

# from pytorch3d.io import IO
from tlspt.io.tls_reader import TLSReader as TR


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
    pc = reader.load_pointcloud(
        path=os.path.join(input_folder, ply_files[0]), device="cpu"
    )

    print(pc.points_packed().shape)
    print(pc.points_packed()[0:3])
    print("  ")
    print(pc.features_packed().shape)
    print(pc.features_packed()[0:3])
    print(pc._feature_names)
    print("====================================")
    print("  ")

    # Save to test.ply
    reader.save_pointcloud(
        pc,
        os.path.join(output_folder, "test.ply"),
        remove=["red", "green", "blue", "scalar_distance"],
        colors_as_uint8=True,
    )

    # Load and print to check
    pc2 = reader.load_pointcloud(
        path=os.path.join(output_folder, "test.ply"), device="cpu"
    )
    print(pc2.points_packed().shape)
    print(pc2.points_packed()[0:3])
    print("  ")
    print(pc2.features_packed().shape)
    print(pc2.features_packed()[0:3])
    print(pc2._feature_names)
    print("  ")


if __name__ == "__main__":
    main()
