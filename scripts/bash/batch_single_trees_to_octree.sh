#!/bin/bash
set -x

base_dir="/home/matt/work/tlsPT/data/plot_octrees"
sites=("allen-spain" "allen-finland" "allen-poland")

for site in "${sites[@]}"; do
  input_base="$base_dir/$site/single_trees"
  output_base="$base_dir/$site/octrees"

  echo "Base directory: $base_dir"
  echo "Processing site: $site"
  echo "Input base: $input_base"

  if [ ! -d "$input_base" ]; then
    echo "Input base directory does not exist: $input_base"
    continue
  fi

  find "$input_base" -mindepth 1 -type d | while read -r input_folder; do
    plot_name=$(basename "$input_folder")

    relative_path="${input_folder#$input_base/}"
    output_folder="$output_base/${relative_path}_1,5m"
    octree_fname="$plot_name.json"
    echo "Input folder: $input_folder"
    echo "Output folder: $output_folder"
    echo "Octree file name: $octree_fname"

    echo "Processing site: $site, Plot: $plot_name"

    python scripts/single_trees_to_octree.py --input_folder "$input_folder" --output_folder "$output_folder" --octree-fname "$octree_fname" --min_scale 1.5
  done
done
