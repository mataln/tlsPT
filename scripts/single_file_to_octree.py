# Script to convert a folder of individual tree point clouds (.ply) to an octree for that plot

from __future__ import annotations

import argparse
import os
import time

from loguru import logger

# from pytorch3d.io import IO
from tlspt.io.tls_reader import TLSReader as TR
from tlspt.structures.file_octree_parallel import FileOctree as FOctree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--octree-fname", type=str, default="octree.json")
    parser.add_argument("--min_scale", type=float, default=2.0)
    parser.add_argument("--hdf5", action="store_true")
    parser.add_argument("--v", action="store_true")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    octree_fname = args.octree_fname
    min_scale = args.min_scale

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

    # Feature renaming
    # name map current -> new
    feature_name_map = {
        "truth": "scalar_truth",
        "label": "scalar_truth",
        "scalar_label": "scalar_truth",
        "distance": "scalar_distance",
        "pathlength": "scalar_pathlength",
        "reflectance": "scalar_reflectance",
        "treeid": "scalar_treeid",
    }
    for pc in pointclouds:
        for i, feature_name in enumerate(pc._feature_names):
            if feature_name in feature_name_map:
                pc._feature_names[i] = feature_name_map[feature_name]

    logger.info("Creating octree")
    start = time.time()
    octree = FOctree(
        pointclouds, min_scale=min_scale
    )  # Scale of leaves will be in min_scale to 2*min_scale
    octree.save(out_folder=output_folder, out_fname=octree_fname)
    print(f"Time taken: {time.time() - start}")

    logger.info("Saving octree voxels")
    out_folder = (
        os.path.join(output_folder, "voxels_hdf5")
        if args.hdf5
        else os.path.join(output_folder, "voxels")
    )
    os.makedirs(out_folder, exist_ok=True)
    octree.save_voxels(
        pointclouds,
        out_folder=out_folder,
        insert_missing_features=True,
        use_hdf5=args.hdf5,
    )

    logger.info(f"Octree voxels saved to {output_folder}")

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
