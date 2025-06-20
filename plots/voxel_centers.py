#!/usr/bin/env python3
"""
Script to visualize a random voxel's full point cloud and its centers
Shows unlabelled data with minimal annotations
Full cloud uses gradient coloring based on z-coordinate (height)

Creates:
1. Full point cloud visualization
2. Centers overlaid on light grey background points with gradient
3. Point cloud with 2D dashed circles overlaid randomly positioned
4. Circular patches extracted from the 2D rendered image under each circle

Features:
- 9 randomly positioned circles by default
- Centers plot shows original points in very light grey with gradient
- Shows split information (train/val/test) unless --publication is used
- Patches are extracted from the actual 2D visualization

Usage:
    python visualize_voxel_centers.py              # Shows index and split
    python visualize_voxel_centers.py --publication # Hides index and split for cleaner figures
    python visualize_voxel_centers.py --idx 12345  # Specific voxel
    python visualize_voxel_centers.py --num-circles 5  # Custom number of circles
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from tlspt.datamodules.components.merged_dataset import MergedOctreeDataset
from tlspt.models.pointmae.components import Group


def setup_3d_plot():
    """Setup a clean 3D plot with good viewing angle"""
    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=45)

    # Hide everything except the points
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    # Make panes invisible
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    # Hide the axes lines
    ax.xaxis.line.set_color("none")
    ax.yaxis.line.set_color("none")
    ax.zaxis.line.set_color("none")

    return fig, ax


def plot_voxel_points(ax, points, voxel_idx=None, split_info=None, show_index=True):
    """Plot points with gradient coloring based on z-coordinate"""
    # Convert to numpy if needed
    if torch.is_tensor(points):
        points = points.cpu().numpy()

    # Use z-coordinate for gradient coloring
    z_vals = points[:, 2]

    # Plot points with viridis colormap
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=z_vals,
        cmap="viridis",
        s=0.8,
        alpha=0.8,
    )

    # Add text info if requested
    if show_index:
        y_pos = 0.95
        if voxel_idx is not None:
            ax.text2D(
                0.05,
                y_pos,
                f"idx: {voxel_idx}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
            )
            y_pos -= 0.05
        if split_info is not None:
            ax.text2D(
                0.05,
                y_pos,
                f"split: {split_info}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    return scatter


def plot_centers(
    ax, centers, points=None, voxel_idx=None, split_info=None, show_index=True
):
    """Plot center points with optional background points"""
    # Plot background points first if provided
    if points is not None:
        if torch.is_tensor(points):
            points = points.cpu().numpy()

        # Use z-coordinate for gradient coloring
        z_vals = points[:, 2]

        # Plot with very light grey gradient
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=z_vals,
            cmap="Greys",
            s=0.3,
            alpha=0.2,
            vmin=z_vals.min(),
            vmax=z_vals.max() * 0.7,
            zorder=1,  # Ensure background is behind
        )

    # Convert centers to numpy if needed
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()

    # Plot centers as larger red points on top
    scatter = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c="red",
        s=50,
        alpha=1.0,
        marker="o",
        edgecolors="black",
        linewidth=0.5,
        zorder=10,  # Ensure centers are on top
    )

    # Add text info if requested
    if show_index:
        y_pos = 0.95
        if voxel_idx is not None:
            ax.text2D(
                0.05,
                y_pos,
                f"idx: {voxel_idx}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
            )
            y_pos -= 0.05
        if split_info is not None:
            ax.text2D(
                0.05,
                y_pos,
                f"split: {split_info}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    return scatter


def calculate_circle_overlap(x1, y1, x2, y2, r):
    """Check if two circles overlap by more than allowed amount"""
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # If circles don't overlap at all
    if dist >= 2 * r:
        return False

    # For controlled overlap: allow circles to overlap by up to 33%
    # This means centers should be at least 1.34*r apart
    # (2r - 1.34r = 0.66r overlap on diameter = ~33% area overlap)
    min_allowed_dist = 1.5 * r

    return dist < min_allowed_dist


def add_circles_to_image(image_path, output_path, num_circles=9):
    """Add dashed circles to a saved image with controlled overlap"""
    # Load the image
    img = Image.open(image_path)
    width, height = img.size

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Limit to middle 75% of image
    xmin = int(width * 0.125)  # 12.5%
    xmax = int(width * 0.875)  # 87.5%
    ymin = int(height * 0.125)  # 12.5%
    ymax = int(height * 0.875)  # 87.5%

    # Define circle positions randomly within the center region
    positions = []
    circle_radius = int(min(xmax - xmin, ymax - ymin) * 0.15)  # 15% of center region

    # Keep circles within bounds
    margin = circle_radius
    valid_xmin = xmin + margin
    valid_xmax = xmax - margin
    valid_ymin = ymin + margin
    valid_ymax = ymax - margin

    # Place circles iteratively
    max_attempts_per_circle = 100

    for i in range(num_circles):
        placed = False
        attempts = 0

        while not placed and attempts < max_attempts_per_circle:
            # Random position within center region
            x = np.random.randint(valid_xmin, valid_xmax)
            y = np.random.randint(valid_ymin, valid_ymax)

            # Check overlap with existing circles
            valid = True
            for px, py, pr in positions:
                if calculate_circle_overlap(x, y, px, py, circle_radius):
                    valid = False
                    break

            if valid or (
                attempts > max_attempts_per_circle // 2
                and len(positions) < num_circles // 2
            ):
                # Place it if valid, or if we're struggling and need to allow some overlap
                positions.append((x, y, circle_radius))
                placed = True

            attempts += 1

        if not placed:
            # If we still can't place it, just place it randomly with overlap allowed
            x = np.random.randint(valid_xmin, valid_xmax)
            y = np.random.randint(valid_ymin, valid_ymax)
            positions.append((x, y, circle_radius))

    print(f"Successfully placed {len(positions)} circles")

    # Draw dashed circles with doubled width
    for x, y, r in positions:
        # Draw circle as a series of arcs to create dashed effect
        num_dashes = 16
        dash_length = 360 / (num_dashes * 2)

        for i in range(num_dashes):
            start_angle = i * dash_length * 2
            end_angle = start_angle + dash_length
            # Draw arc segments with doubled width
            bbox = [x - r, y - r, x + r, y + r]
            draw.arc(bbox, start=start_angle, end=end_angle, fill="black", width=6)

    # Save the image with circles
    img.save(output_path, dpi=(300, 300))

    return positions


def extract_circular_patches(image_path, circle_positions, output_dir, voxel_idx):
    """Extract circular regions from the base image (without circles drawn on it)"""
    # Load the image
    img = Image.open(image_path)

    for i, (x, y, r) in enumerate(circle_positions):
        # Create a square crop around the circle
        left = max(0, x - r)
        top = max(0, y - r)
        right = min(img.width, x + r)
        bottom = min(img.height, y + r)

        # Crop the region
        patch = img.crop((left, top, right, bottom))

        # Create a circular mask
        mask = Image.new("L", (right - left, bottom - top), 0)
        draw = ImageDraw.Draw(mask)

        # Draw a filled circle on the mask
        circle_center_x = x - left
        circle_center_y = y - top
        draw.ellipse(
            [
                circle_center_x - r,
                circle_center_y - r,
                circle_center_x + r,
                circle_center_y + r,
            ],
            fill=255,
        )

        # Apply the mask to get circular patch
        patch.putalpha(mask)

        # Save on white background
        final_patch = Image.new("RGBA", patch.size, (255, 255, 255, 255))
        final_patch.paste(patch, (0, 0), patch)
        final_patch = final_patch.convert("RGB")

        # Save the patch
        patch_path = os.path.join(output_dir, f"voxel_{voxel_idx}_patch_{i}.png")
        final_patch.save(patch_path, dpi=(300, 300))
        print(f"Saved patch {i}: {patch_path}")


def plot_voxel_simple(ax, points, voxel_idx=None, split_info=None, show_index=True):
    """Plot points with gradient coloring - no circles"""
    # Convert to numpy if needed
    if torch.is_tensor(points):
        points = points.cpu().numpy()

    # Use z-coordinate for gradient coloring
    z_vals = points[:, 2]

    # Plot all points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=z_vals,
        cmap="viridis",
        s=0.8,
        alpha=0.8,
    )

    # Add text info if requested
    if show_index:
        y_pos = 0.95
        if voxel_idx is not None:
            ax.text2D(
                0.05,
                y_pos,
                f"idx: {voxel_idx}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
            )
            y_pos -= 0.05
        if split_info is not None:
            ax.text2D(
                0.05,
                y_pos,
                f"split: {split_info}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])


def visualize_voxel(
    voxel_idx=None, output_dir="saved_plots", show_index=True, num_circles=9
):
    """Load and visualize a voxel showing full cloud and centers

    Args:
        voxel_idx: Specific voxel index to visualize (if None, selects randomly)
        output_dir: Base directory to save outputs
        show_index: Whether to show the voxel index on plots
        num_circles: Number of circles/patches to extract
    """

    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define the data paths for UNLABELLED supertree data
    split_files = [
        "../data/supertree/plot_octrees_1cm/hjfo-fin/hjfo-fin-splits.csv",
        "../data/supertree/plot_octrees_1cm/hjfo-pol/hjfo-pol-splits.csv",
        "../data/supertree/plot_octrees_1cm/hjfo-spa/hjfo-spa-splits.csv",
        "../data/supertree/plot_octrees_1cm/widi-aus/widi-aus-splits.csv",
    ]

    # Find available split files
    available_splits = []
    for split_file in split_files:
        if os.path.exists(split_file):
            available_splits.append(split_file)
            print(f"Found split file: {split_file}")

    if not available_splits:
        print("No split files found. Please check your data directory.")
        return

    # Create merged dataset from all available split files
    print(f"\nLoading merged dataset from {len(available_splits)} split files")

    dataset = MergedOctreeDataset(
        split_files=available_splits,
        split="train",  # Use training split for visualization
        scales=2,  # Same scale for all datasets
        feature_names=None,  # No features - unlabelled
        normalize=True,
        transform=None,
        min_points=512,
    )

    print(f"Dataset size: {len(dataset)} voxels")

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    # Select random voxel if not specified
    if voxel_idx is None:
        voxel_idx = np.random.randint(0, len(dataset))
        print(f"Randomly selected voxel index: {voxel_idx}")

    if voxel_idx >= len(dataset):
        print(f"Voxel index {voxel_idx} is out of range (max: {len(dataset)-1})")
        return

    # Create voxel-specific output directory
    voxel_output_dir = os.path.join(output_dir, f"voxel_{voxel_idx}")
    os.makedirs(voxel_output_dir, exist_ok=True)
    print(f"\nSaving all outputs to: {voxel_output_dir}")

    # Get plot name and split info for the specific voxel
    dataset_idx, local_idx = dataset.idxs[voxel_idx]
    hdf5_path, _, _ = dataset.datasets[dataset_idx].voxels_to_load[local_idx]
    plot_name = os.path.basename(os.path.dirname(hdf5_path))

    # Get split info
    split_info = dataset.datasets[dataset_idx].split if show_index else None

    # Load the specific voxel data
    print(f"\nLoading voxel {voxel_idx} from plot {plot_name} (split: {split_info})")
    try:
        data = dataset[voxel_idx]
        points = data["points"]
        scale = data.get("scales", 2)
        lengths = data["lengths"]

        print(f"  Points shape: {points.shape}")
        print(f"  Number of points: {lengths}")
        print(f"  Scale: {scale}m")

        # 1. Create full point cloud visualization
        fig1, ax1 = setup_3d_plot()
        plot_voxel_points(
            ax1,
            points,
            voxel_idx=voxel_idx,
            split_info=split_info,
            show_index=show_index,
        )

        # Save full cloud visualization
        full_cloud_path = os.path.join(
            voxel_output_dir, f"voxel_{voxel_idx}_full_cloud.png"
        )
        plt.savefig(full_cloud_path, dpi=300, bbox_inches="tight")
        print(
            f"Saved full cloud visualization (gradient colored by height): {full_cloud_path}"
        )
        plt.close(fig1)

        # 2. Create centers visualization with background points
        # Use the Group module to get centers
        print("\nComputing centers...")
        group = Group(
            num_centers=64,  # You can adjust this
            num_neighbors=32,  # You can adjust this
            neighbor_alg="ball_query",
            radius=0.1 / scale,  # Adjust radius based on scale
        )

        # Group expects batch dimension
        points_batch = points.unsqueeze(0)  # Add batch dimension
        lengths_batch = torch.tensor([lengths])

        patches, centers = group(points_batch, lengths_batch)
        centers = centers.squeeze(0)  # Remove batch dimension

        print(f"  Centers shape: {centers.shape}")
        print(f"  Number of centers: {centers.shape[0]}")

        fig2, ax2 = setup_3d_plot()
        plot_centers(
            ax2,
            centers,
            points=points,
            voxel_idx=voxel_idx,
            split_info=split_info,
            show_index=show_index,
        )

        # Save centers visualization
        centers_path = os.path.join(voxel_output_dir, f"voxel_{voxel_idx}_centers.png")
        plt.savefig(centers_path, dpi=300, bbox_inches="tight")
        print(f"Saved centers visualization: {centers_path}")
        plt.close(fig2)

        # 3. Create visualization without circles first
        print("\nCreating base visualization...")

        fig3, ax3 = setup_3d_plot()
        plot_voxel_simple(
            ax3,
            points,
            voxel_idx=voxel_idx,
            split_info=split_info,
            show_index=show_index,
        )

        # Save base visualization without circles
        base_path = os.path.join(voxel_output_dir, f"voxel_{voxel_idx}_base.png")
        plt.savefig(base_path, dpi=300, bbox_inches="tight")
        plt.close(fig3)

        # 4. Add circles to the saved image
        print("\nAdding circles to visualization...")
        circles_path = os.path.join(
            voxel_output_dir, f"voxel_{voxel_idx}_with_circles.png"
        )
        circle_positions = add_circles_to_image(
            base_path, circles_path, num_circles=num_circles
        )
        print(f"Saved visualization with circles: {circles_path}")

        # 5. Extract patches from the base image (without circles)
        print(
            "\nExtracting patches from visualization (circles not included in patches)..."
        )
        extract_circular_patches(
            base_path, circle_positions, voxel_output_dir, voxel_idx
        )

        # Clean up temporary base file
        os.remove(base_path)

    except Exception as e:
        print(f"Error processing voxel {voxel_idx}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize TLS voxel and its centers")
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Publication mode - hide index values for cleaner figures",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Specific voxel index to visualize (if not specified, selects randomly)",
    )
    parser.add_argument(
        "--num-circles",
        type=int,
        default=9,
        help="Number of circles/patches to extract (default: 9)",
    )
    args = parser.parse_args()

    print("Starting voxel visualization...")

    # Show index unless in publication mode
    show_index = not args.publication

    visualize_voxel(
        voxel_idx=args.idx,
        output_dir="saved_plots",
        show_index=show_index,
        num_circles=args.num_circles,
    )

    print("\nVisualization complete!")
