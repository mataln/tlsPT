#!/bin/bash

# Base directory
base_dir="/home/matt/work/tlsPT/data/plot_octrees"

# List of site folders
sites=("allen-spain" "allen-finland" "allen-poland")

# Iterate over each site folder
for site in "${sites[@]}"; do
  input_base="$base_dir/$site/single_trees"
  output_base="$base_dir/$site/octrees"

  # Find all subdirectories within the input base directory
  find "$input_base" -type d | while read -r input_folder; do
    # Get the last component of the input folder path (e.g., SPA01, SPA02)
    plot_name=$(basename "$input_folder")

    # Derive the output folder path based on the input folder's path
    relative_path="${input_folder#$input_base/}"
    output_folder="$output_base/${relative_path}_1,5m"

    # Print the current folder and plot name
    echo "Processing site: $site, Plot: $plot_name"

    # Run the Python command with additional parameters
    python scripts/single_trees_to_octree.py --input_folder "$input_folder" --output_folder "$output_folder" --octree-fname "$plot_name" --min_scale 1.5
  done
done
