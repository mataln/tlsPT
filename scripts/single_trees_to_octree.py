#Script to convert a folder of individual tree point clouds (.ply) to an octree for that plot

import os 
import sys
import numpy as np
import argparse

from loguru import logger
#from pytorch3d.io import IO
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
    pc = reader.load_pointcloud(path=os.path.join(input_folder, ply_files[0]), device="cpu")

    print(pc.points_packed().shape)
    print(pc.features_packed().shape)   
    print(pc.features_packed()[0:3])
    print(pc._feature_names)

if __name__ == "__main__":
    main()


