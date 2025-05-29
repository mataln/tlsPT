#!/bin/bash
set -x
base_dir="/rds/project/rds-1ZDMOiwfVqk/tlspt-labelled/plot_clouds" #"/home/mja78/work/tlsPT/data/supertree/plot_clouds_1cm"
output_base="/rds/project/rds-1ZDMOiwfVqk/tlspt-labelled/plot_octrees" #"/home/mja78/work/tlsPT/data/supertree/octrees_1cm"
sites=("hjfo-spa") #hjfo-cam  hjfo-chi  hjfo-fin  hjfo-ger  hjfo-pol  hjfo-spa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for site in "${sites[@]}"; do
  input_base="$base_dir/$site/" #e.g. /home/matt/work/tlsPT/data/plot_octrees/allen-spain/single_trees
  site_output_base="$output_base/$site/" #e.g. /home/matt/work/tlsPT/data/plot_octrees/allen-spain/octrees_2m

  echo "Processing site: $site"
  echo "Input base: $input_base"
  echo "Output base: $site_output_base"

  if [ ! -d "$input_base" ]; then
    echo "Input base directory does not exist: $input_base"
    continue
  fi

  find "$input_base" -mindepth 1 -type d | while read -r input_folder; do
    plot_name=$(basename "$input_folder")  #e.g. SPA01

    # #Temporary override - skip all plots except MLA-01 for testing:
    # if [ "$plot_name" != "MLA-01" ]; then
    #   echo "Skipping plot: $plot_name"
    #   continue
    # fi

    output_folder="$site_output_base/${plot_name}_2m" #e.g. /home/mja78/work/tlsPT/data/supertree/octrees_1cm/hjfo-fin/SPA01_2m
    octree_fname="$plot_name.json"
    voxels_fname="$plot_name.h5"

    echo "Input folder: $input_folder"
    echo "Output folder: $output_folder"
    echo "Octree file name: $octree_fname"
    echo "Voxels file name: $voxels_fname"
    echo "Processing site: $site, Plot: $plot_name"
    echo " "

    python scripts/single_file_to_octree.py \
      --input_folder "$input_folder" \
      --output_folder "$output_folder" \
      --octree-fname "$octree_fname" \
      --voxels-fname "$voxels_fname" \
      --min_scale 2.0 \
      --v

    #Exit early if MLA-01 is processed
    # if [ "$plot_name" == "MLA-01" ]; then
    #   echo "Processed MLA-01, exiting early."
    #   exit 0
    # fi
  done
done
