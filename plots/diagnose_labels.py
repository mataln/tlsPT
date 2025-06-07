#!/usr/bin/env python3
"""
Script to visualize voxelized TLS data with leaf/wood labels
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tlspt.datamodules.components.octree_dataset import OctreeDataset
from tlspt.structures.file_octree import FileOctree


def setup_3d_plot(title="Voxel Visualization"):
    """Setup a 3D plot with good viewing angle"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=45)
    return fig, ax


def plot_voxel_points(ax, points, labels, voxel_scale=None):
    """Plot points colored by leaf/wood label"""
    # Convert to numpy if needed
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Squeeze labels if needed
    if labels.ndim > 1:
        labels = labels.squeeze()

    # Create color array (0=wood=brown, 1=leaf=green)
    colors = np.where(labels == 1, "green", "saddlebrown")

    # Plot points
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5, alpha=0.6
    )

    # Add scale info if provided
    if voxel_scale is not None:
        ax.text2D(
            0.05,
            0.95,
            f"Scale: {voxel_scale:.2f}m",
            transform=ax.transAxes,
            fontsize=10,
        )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Leaf"),
        Patch(facecolor="saddlebrown", label="Wood"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    return scatter


def visualize_dataset_samples(output_dir="saved_plots", num_samples=6):
    """Load and visualize samples from the dataset"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define the data paths (based on your config files)
    split_files = [
        "../data/tlspt_labelled/plot_octrees/hjfo-finl/hjfo-finl-splits.csv",
        "../data/tlspt_labelled/plot_octrees/hjfo-poll/hjfo-poll-splits.csv",
        "../data/tlspt_labelled/plot_octrees/hjfo-spal/hjfo-spal-splits.csv",
    ]

    # Try to find which split files exist
    available_splits = []
    for split_file in split_files:
        if os.path.exists(split_file):
            available_splits.append(split_file)
            print(f"Found split file: {split_file}")
        else:
            print(f"Split file not found: {split_file}")

    if not available_splits:
        print("No split files found. Please check your data directory.")
        return

    # Use the first available split file
    split_file = available_splits[0]

    # Create dataset
    print(f"\nLoading dataset from {split_file}")
    dataset = OctreeDataset(
        split_file=split_file,
        split="train",  # Use training split for visualization
        scale=2,
        feature_names=["scalar_truth"],
        features_to_normalize=None,
        normalize=False,  # Don't normalize for visualization
        transform=None,
        min_points=512,
    )

    print(f"Dataset size: {len(dataset)} voxels")

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    # Sample some voxels to visualize
    num_to_plot = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_to_plot, replace=False)

    # Create a multi-page PDF
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = os.path.join(output_dir, "voxel_visualizations.pdf")

    with PdfPages(pdf_path) as pdf:
        for i, idx in enumerate(indices):
            print(f"\nProcessing voxel {i+1}/{num_to_plot} (index {idx})")

            # Load the voxel data
            try:
                data = dataset[idx]
                points = data["points"]
                labels = data["features"]
                scale = data.get("scales", 2)

                print(f"  Points shape: {points.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Unique labels: {torch.unique(labels).cpu().numpy()}")

                # Create the plot
                fig, ax = setup_3d_plot(f"Voxel {idx} - Scale {scale}m")
                plot_voxel_points(ax, points, labels, voxel_scale=scale)

                # Save to PDF
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            except Exception as e:
                print(f"  Error processing voxel {idx}: {e}")
                continue

    print(f"\nVisualization saved to: {pdf_path}")

    # Also create individual plots for the first few
    for i, idx in enumerate(indices[:3]):
        try:
            data = dataset[idx]
            points = data["points"]
            labels = data["features"]
            scale = data.get("scales", 2)

            fig, ax = setup_3d_plot(f"Voxel {idx}")
            plot_voxel_points(ax, points, labels, voxel_scale=scale)

            # Save individual PNG
            png_path = os.path.join(output_dir, f"voxel_{idx}.png")
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved individual plot: {png_path}")

        except Exception as e:
            print(f"Error saving individual plot for voxel {idx}: {e}")


def visualize_octree_structure(output_dir="saved_plots"):
    """Visualize the octree structure itself"""

    # Try to find an octree JSON file
    data_dirs = [
        "../data/tlspt_labelled/plot_octrees/hjfo-finl",
        "../data/tlspt_labelled/plot_octrees/hjfo-poll",
        "../data/tlspt_labelled/plot_octrees/hjfo-spal",
    ]

    json_file = None
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
            if json_files:
                json_file = os.path.join(data_dir, json_files[0])
                break

    if not json_file:
        print("No octree JSON file found")
        return

    print(f"\nLoading octree from: {json_file}")
    octree = FileOctree(json_file)

    # Get nodes at different scales
    nodes_scale_1 = octree.find_nodes_by_scale(1)
    nodes_scale_2 = octree.find_nodes_by_scale(2)
    nodes_scale_4 = octree.find_nodes_by_scale(4)

    print(f"Nodes at scale ~1m: {len(nodes_scale_1)}")
    print(f"Nodes at scale ~2m: {len(nodes_scale_2)}")
    print(f"Nodes at scale ~4m: {len(nodes_scale_4)}")

    # Plot the octree structure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Octree Structure Visualization", fontsize=14, fontweight="bold")

    # Plot nodes at different scales with different colors
    scales_to_plot = [
        (nodes_scale_4[:20], "red", 0.3, "Scale ~4m"),
        (nodes_scale_2[:50], "blue", 0.2, "Scale ~2m"),
        (nodes_scale_1[:100], "green", 0.1, "Scale ~1m"),
    ]

    for nodes, color, alpha, label in scales_to_plot:
        for node in nodes:
            bbox = node.bbox.numpy()
            # Draw a wireframe box for each node
            corners = np.array(
                [
                    [bbox[0, 0], bbox[1, 0], bbox[2, 0]],
                    [bbox[0, 1], bbox[1, 0], bbox[2, 0]],
                    [bbox[0, 1], bbox[1, 1], bbox[2, 0]],
                    [bbox[0, 0], bbox[1, 1], bbox[2, 0]],
                    [bbox[0, 0], bbox[1, 0], bbox[2, 1]],
                    [bbox[0, 1], bbox[1, 0], bbox[2, 1]],
                    [bbox[0, 1], bbox[1, 1], bbox[2, 1]],
                    [bbox[0, 0], bbox[1, 1], bbox[2, 1]],
                ]
            )

            # Draw edges
            edges = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # bottom
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # top
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # vertical
            ]

            for edge in edges:
                points = corners[edge]
                ax.plot3D(*points.T, color=color, alpha=alpha, linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color, label=label) for _, color, _, label in scales_to_plot
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Save
    pdf_path = os.path.join(output_dir, "octree_structure.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Octree structure saved to: {pdf_path}")


if __name__ == "__main__":
    print("Starting voxel visualization...")

    # Visualize voxel data
    visualize_dataset_samples(output_dir="saved_plots", num_samples=6)

    # Visualize octree structure
    visualize_octree_structure(output_dir="saved_plots")

    print("\nVisualization complete!")
