#!/usr/bin/env python3
"""
Script to visualize voxelized TLS data with reconstruction from trained PointMAE model

Creates a grid visualization with 4 columns:
1. Original: Original voxel data (gradient colored by height for unlabeled, leaf/wood colors for labeled)
2. Subsampled: Subsampled point cloud (all patches before masking)
3. Masked: Voxel with masked patches removed (showing only visible patches)
4. Reconstructed: Full reconstruction (visible patches + reconstructed masked patches)

For labeled data:
- Leaves (label 0) = dark grey
- Wood (label 1) = red
- Reconstructed patches = blue

Uses validation split for better coverage of the data.
All visualizations are shown in a single grid with:
- Column headers at the top
- Plot names as row labels on the left
- Consistent axis limits and color gradients within each row

UPDATES:
- Only includes plots where the voxel has <= 33000 points
- Removes tick labels from all axes
- Shows voxel index on each plot for reproducibility
- Can specify exact voxel indices to visualize
- Removes "_2m" suffix from plot names
- Ensures consistent color gradient within each row
- Creates a single grid visualization instead of multiple pages
- Added "Subsampled" column showing patches before masking
- Optional --publication mode to hide index values for cleaner figures

Usage examples:
    # Random selection
    python reconstruction_grid.py

    # Specific voxel indices
    python reconstruction_grid.py --indices 15 42 108 234

    # Custom parameters
    python reconstruction_grid.py --num-samples 10 --max-points 50000

    # Publication mode (hide index values)
    python reconstruction_grid.py --publication

Save as: plots/reconstruction_grid.py
"""

from __future__ import annotations

import argparse
import os
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # Important for 3D plotting
from omegaconf import OmegaConf

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from tlspt.datamodules.components.merged_dataset import MergedOctreeDataset
from tlspt.models.utils import get_masked


def clean_plot_name(plot_name):
    """Remove common suffixes from plot names"""
    # Remove _2m suffix if present
    if plot_name.endswith("_2m"):
        plot_name = plot_name[:-3]
    return plot_name


def remove_tick_labels(ax):
    """Remove tick labels from all axes"""
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def setup_3d_plot(title="Voxel Visualization", figsize=(8, 6)):
    """Setup a 3D plot with good viewing angle"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=45)
    remove_tick_labels(ax)
    return fig, ax


def get_axis_limits(points_list):
    """Calculate axis limits to encompass all points in the list"""
    all_points = []
    for points in points_list:
        if torch.is_tensor(points):
            points = points.cpu().numpy()
        if len(points) > 0:  # Only add non-empty arrays
            all_points.append(points)

    if not all_points:
        return ((-1, 1), (-1, 1), (-1, 1))  # Default if no points

    all_points = np.vstack(all_points)

    # Calculate limits with some padding
    padding = 0.05
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Handle case where range is zero
    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1
    if z_range == 0:
        z_range = 1

    return (
        (x_min - padding * x_range, x_max + padding * x_range),
        (y_min - padding * y_range, y_max + padding * y_range),
        (z_min - padding * z_range, z_max + padding * z_range),
    )


def get_color_limits(points_list):
    """Calculate consistent color limits (z-value range) for all points in the list"""
    all_z_values = []
    for points in points_list:
        if torch.is_tensor(points):
            points = points.cpu().numpy()
        if len(points) > 0:
            all_z_values.extend(points[:, 2])

    if not all_z_values:
        return 0, 1  # Default if no points

    return min(all_z_values), max(all_z_values)


def apply_axis_limits(ax, xlim, ylim, zlim):
    """Apply axis limits to a 3D plot"""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


def plot_original_voxel(ax, points, title="", vmin=None, vmax=None):
    """Plot original voxel with gradient coloring"""
    # Convert to numpy if needed
    if torch.is_tensor(points):
        points = points.cpu().numpy()

    # Use z-coordinate for gradient coloring
    z_vals = points[:, 2]

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=z_vals,
        cmap="viridis",
        s=0.8,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    remove_tick_labels(ax)


def plot_subsampled_patches(ax, patches, centers, title="", vmin=None, vmax=None):
    """Plot all patches (subsampled point cloud before masking)"""
    # Convert to numpy if needed
    if torch.is_tensor(patches):
        patches = patches.cpu().numpy()
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()

    # Plot all patches
    all_points = []
    for i in range(patches.shape[0]):
        patch_points = patches[i] + centers[i]
        all_points.append(patch_points)

    if all_points:
        all_points = np.vstack(all_points)
        z_vals = all_points[:, 2]
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            c=z_vals,
            cmap="viridis",
            s=0.8,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    remove_tick_labels(ax)


def plot_visible_patches_only(
    ax, patches, centers, mask, title="", vmin=None, vmax=None
):
    """Plot only the visible (unmasked) patches"""
    # Convert to numpy if needed
    if torch.is_tensor(patches):
        patches = patches.cpu().numpy()
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    # Get visible patches (where mask is False)
    vis_patches = patches[~mask]
    vis_centers = centers[~mask]

    # Plot each visible patch
    all_points = []
    for i in range(vis_patches.shape[0]):
        patch_points = vis_patches[i] + vis_centers[i]
        all_points.append(patch_points)

    if all_points:
        all_points = np.vstack(all_points)
        z_vals = all_points[:, 2]
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            c=z_vals,
            cmap="viridis",
            s=0.8,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    remove_tick_labels(ax)


def plot_full_reconstruction(
    ax,
    patches,
    centers,
    mask,
    reconstructed_patches,
    masked_centers,
    title="",
    vmin=None,
    vmax=None,
):
    """Plot full reconstruction with visible + reconstructed patches"""
    # Convert to numpy if needed
    if torch.is_tensor(patches):
        patches = patches.cpu().numpy()
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if torch.is_tensor(reconstructed_patches):
        reconstructed_patches = reconstructed_patches.cpu().numpy()
    if torch.is_tensor(masked_centers):
        masked_centers = masked_centers.cpu().numpy()

    # Collect all points
    all_points = []

    # Add visible patches
    vis_patches = patches[~mask]
    vis_centers = centers[~mask]
    for i in range(vis_patches.shape[0]):
        patch_points = vis_patches[i] + vis_centers[i]
        all_points.append(patch_points)

    # Add reconstructed patches
    for i in range(reconstructed_patches.shape[0]):
        patch_points = reconstructed_patches[i] + masked_centers[i]
        all_points.append(patch_points)

    if all_points:
        all_points = np.vstack(all_points)
        z_vals = all_points[:, 2]
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            c=z_vals,
            cmap="viridis",
            s=0.8,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    remove_tick_labels(ax)


def plot_labeled_original(ax, points, labels, title=""):
    """Plot original voxel with leaf/wood labels"""
    # Convert to numpy if needed
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Create colors based on labels (0=leaf=dark grey, 1=wood=red)
    colors = np.where(labels == 0, "dimgrey", "red")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.8, alpha=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    remove_tick_labels(ax)

    # Add legend
    legend_elements = [
        Patch(facecolor="dimgrey", label="Leaf"),
        Patch(facecolor="red", label="Wood"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")


def plot_labeled_subsampled(ax, patches, centers, points, labels, title=""):
    """Plot all patches (subsampled) with correct labels"""
    # Convert to numpy if needed
    if torch.is_tensor(patches):
        patches = patches.cpu().numpy()
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # For each patch, find which points belong to it and get their labels
    all_points = []
    all_colors = []

    for i in range(patches.shape[0]):
        patch_points = patches[i] + centers[i]

        # Find closest original points to get their labels
        patch_center = centers[i]
        distances = np.linalg.norm(points - patch_center, axis=1)
        closest_indices = np.argsort(distances)[: patches.shape[1]]
        patch_labels = labels[closest_indices]
        patch_colors = np.where(patch_labels == 0, "dimgrey", "red")

        all_points.append(patch_points)
        all_colors.extend(patch_colors)

    if all_points:
        all_points = np.vstack(all_points)
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            c=all_colors,
            s=0.8,
            alpha=0.8,
        )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    remove_tick_labels(ax)


def plot_labeled_visible_only(ax, patches, centers, mask, points, labels, title=""):
    """Plot only visible patches with correct labels"""
    # Convert to numpy if needed
    if torch.is_tensor(patches):
        patches = patches.cpu().numpy()
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Get visible patches (where mask is False)
    vis_patches = patches[~mask]
    vis_centers = centers[~mask]

    # For each visible patch, we need to find which points belong to it
    # and get their labels
    all_points = []
    all_colors = []

    for i in range(vis_patches.shape[0]):
        patch_points = vis_patches[i] + vis_centers[i]

        # Find closest original points to get their labels
        # This is approximate but should work for visualization
        patch_center = vis_centers[i]
        distances = np.linalg.norm(points - patch_center, axis=1)
        closest_indices = np.argsort(distances)[: vis_patches.shape[1]]
        patch_labels = labels[closest_indices]
        patch_colors = np.where(patch_labels == 0, "dimgrey", "red")

        all_points.append(patch_points)
        all_colors.extend(patch_colors)

    if all_points:
        all_points = np.vstack(all_points)
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            c=all_colors,
            s=0.8,
            alpha=0.8,
        )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    remove_tick_labels(ax)


def plot_labeled_reconstruction(
    ax,
    patches,
    centers,
    mask,
    reconstructed_patches,
    masked_centers,
    points,
    labels,
    title="",
):
    """Plot full reconstruction maintaining original labels for visible patches"""
    # Convert everything to numpy
    if torch.is_tensor(patches):
        patches = patches.cpu().numpy()
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if torch.is_tensor(reconstructed_patches):
        reconstructed_patches = reconstructed_patches.cpu().numpy()
    if torch.is_tensor(masked_centers):
        masked_centers = masked_centers.cpu().numpy()
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Visible patches with original labels
    vis_patches = patches[~mask]
    vis_centers = centers[~mask]

    all_points = []
    all_colors = []

    # Add visible patches with their original labels
    for i in range(vis_patches.shape[0]):
        patch_points = vis_patches[i] + vis_centers[i]

        # Find labels for this patch
        patch_center = vis_centers[i]
        distances = np.linalg.norm(points - patch_center, axis=1)
        closest_indices = np.argsort(distances)[: vis_patches.shape[1]]
        patch_labels = labels[closest_indices]
        patch_colors = np.where(patch_labels == 0, "dimgrey", "red")

        all_points.append(patch_points)
        all_colors.extend(patch_colors)

    # Add reconstructed patches in blue to show they're reconstructed
    for i in range(reconstructed_patches.shape[0]):
        patch_points = reconstructed_patches[i] + masked_centers[i]
        all_points.append(patch_points)
        all_colors.extend(["blue"] * patch_points.shape[0])

    if all_points:
        all_points = np.vstack(all_points)
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            c=all_colors,
            s=0.8,
            alpha=0.8,
        )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    remove_tick_labels(ax)

    # Legend
    legend_elements = [
        Patch(facecolor="dimgrey", label="Leaf"),
        Patch(facecolor="red", label="Wood"),
        Patch(facecolor="blue", label="Reconstructed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")


def visualize_reconstructions(
    checkpoint_path="/home/mja78/work/tlsPT/checkpoints_saved/vits_radius0.2_neighbors32_mr07_uldata1.0_best_loss0.0018.ckpt",
    config_path="../configs/pretrain/pretrain_vits_mr07.yaml",
    output_dir="saved_plots",
    num_samples=6,
    max_points=33000,
    specific_indices=None,  # List of specific voxel indices to visualize
    publication_mode=False,  # Whether to hide index values
):
    """Load model and visualize reconstructions

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config
        output_dir: Directory to save outputs
        num_samples: Number of samples to visualize (if not using specific_indices)
        max_points: Maximum points per voxel
        specific_indices: List of specific voxel indices to visualize
        publication_mode: If True, hide index values in labels for cleaner publication figures
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    print(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)

    # Manually resolve config references since we're not using full Hydra
    # The transencoder and transdecoder configs reference ${model.embedding_dim}
    embedding_dim = config.model.embedding_dim
    config.model.transencoder_config.embed_dim = embedding_dim
    config.model.transdecoder_config.embed_dim = embedding_dim

    # total_epochs references ${max_epochs}
    config.model.total_epochs = config.max_epochs

    # Initialize model from config using Hydra utils
    print("Initializing model from config")
    model = hydra.utils.instantiate(config.model)

    # Load checkpoint weights
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load model weights
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print("Model loaded successfully")

    # Define the data paths
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
            # Don't break - we want ALL split files!

    if not available_splits:
        print("No split files found. Please check your data directory.")
        return

    # Create merged dataset from all available split files
    print(f"\nLoading merged dataset from {len(available_splits)} split files")

    dataset = MergedOctreeDataset(
        split_files=available_splits,
        split="val",  # Use validation split
        scales=2,  # Same scale for all datasets
        feature_names=None,  # No features needed for reconstruction
        normalize=True,
        transform=None,
        min_points=512,
    )

    print(f"Dataset size: {len(dataset)} voxels")

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    # Sample voxels but filter by point count
    if specific_indices:
        # Use specific indices if provided
        indices = specific_indices
        print(f"\nUsing specific voxel indices: {indices}")

        # Verify these indices are valid and meet criteria
        valid_indices = []
        for idx in indices:
            if idx >= len(dataset):
                print(
                    f"Warning: Index {idx} is out of range (dataset size: {len(dataset)})"
                )
                continue
            try:
                data = dataset[idx]
                points = data["points"]
                if points.shape[0] > max_points:
                    print(
                        f"Warning: Voxel {idx} has {points.shape[0]} points (> {max_points}), skipping"
                    )
                    continue
                valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: Could not load voxel {idx}: {e}")
                continue

        indices = valid_indices
        if len(indices) == 0:
            print("No valid indices found!")
            return
    else:
        # Filter by point count and randomly sample
        valid_indices = []
        for idx in range(len(dataset)):
            try:
                data = dataset[idx]
                points = data["points"]
                if points.shape[0] <= max_points:
                    valid_indices.append(idx)
            except:
                continue

        print(f"Found {len(valid_indices)} voxels with <= {max_points} points")

        if len(valid_indices) == 0:
            print(f"No voxels found with <= {max_points} points!")
            return

        # Sample from valid indices
        num_to_plot = min(num_samples, len(valid_indices))
        indices = np.random.choice(valid_indices, num_to_plot, replace=False)
        print(f"Randomly selected indices: {sorted(indices.tolist())}")

    # First, process all voxels to collect the data
    all_voxel_data = []

    for i, idx in enumerate(indices):
        # Get plot name first
        dataset_idx, local_idx = dataset.idxs[idx]
        hdf5_path, _, _ = dataset.datasets[dataset_idx].voxels_to_load[local_idx]
        plot_name = os.path.basename(os.path.dirname(hdf5_path))
        plot_name = clean_plot_name(plot_name)  # Clean the plot name
        print(
            f"\nProcessing voxel {i+1}/{len(indices)} - Index: {idx} from plot {plot_name}"
        )

        try:
            # Load voxel data
            data = dataset[idx]
            points = data["points"]
            lengths = data["lengths"]

            # Print number of points in original voxel
            print(f"Original voxel has {points.shape[0]} points")

            # Double-check point count
            if points.shape[0] > max_points:
                print(f"Skipping voxel with {points.shape[0]} points (> {max_points})")
                continue

            # Prepare input for model
            batch = {
                "points": points.unsqueeze(0),  # Add batch dimension
                "lengths": torch.tensor([lengths]),
            }

            with torch.no_grad():
                # Get patches and centers
                patches, centers = model.group(batch["points"], batch["lengths"])

                # Get mask
                mask = model.mask_generator(centers)

                # Get masked patches for reconstruction
                get_masked(patches, mask)
                masked_centers = get_masked(centers, mask)

                # Run encoder
                x_vis, _, vis_pos_embeddings = model.forward_encoder(patches, centers)

                # Prepare decoder input
                masked_pos_embeddings = model.pos_encoder(masked_centers)
                B, N, _ = masked_pos_embeddings.shape
                mask_tokens = model.mask_token.expand(B, N, -1)

                x_full = torch.cat((x_vis, mask_tokens), dim=1)
                full_pos_embeddings = torch.cat(
                    (vis_pos_embeddings, masked_pos_embeddings), dim=1
                )

                if model.encoder_to_decoder_proj is not None:
                    x_full = model.encoder_to_decoder_proj(x_full)
                    full_pos_embeddings = model.encoder_to_decoder_proj(
                        full_pos_embeddings
                    )

                # Decode to get reconstructed patches
                x_hat = model.forward_decoder(x_full, full_pos_embeddings, N)

            # Store all the data for this voxel
            voxel_data = {
                "plot_name": plot_name,
                "idx": idx,
                "points": points.squeeze(0).cpu().numpy(),
                "patches": patches.squeeze(0).cpu().numpy(),
                "centers": centers.squeeze(0).cpu().numpy(),
                "mask": mask.squeeze(0).cpu().numpy(),
                "x_hat": x_hat.squeeze(0).cpu().numpy(),
                "masked_centers": masked_centers.squeeze(0).cpu().numpy(),
                "num_points": points.shape[0],
                "num_masked": mask.sum().item(),
            }

            all_voxel_data.append(voxel_data)

        except Exception as e:
            print(f"Error processing voxel from {plot_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Now create the grid visualization
    n_rows = len(all_voxel_data)
    n_cols = 4  # Original, Subsampled, Masked, Reconstructed

    # Create figure with appropriate size
    fig_width = 4.5 * n_cols  # Slightly wider for better spacing
    fig_height = 4 * n_rows + 0.5  # Less extra space needed for titles
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Add column titles - properly aligned with subplot centers
    col_titles = ["Original", "Subsampled", "Masked", "Reconstructed"]
    left_margin = 0.15
    right_margin = 0.98
    plot_width = right_margin - left_margin
    for j, title in enumerate(col_titles):
        # Calculate position to center over each column accounting for margins
        col_center = left_margin + (j + 0.5) * plot_width / n_cols
        fig.text(
            col_center,
            0.98,
            title,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    # Process each voxel
    for i, voxel_data in enumerate(all_voxel_data):
        # Extract data
        plot_name = voxel_data["plot_name"]
        idx = voxel_data["idx"]
        points = voxel_data["points"]
        patches = voxel_data["patches"]
        centers = voxel_data["centers"]
        mask = voxel_data["mask"]
        x_hat = voxel_data["x_hat"]
        masked_centers = voxel_data["masked_centers"]
        voxel_data["num_points"]
        voxel_data["num_masked"]

        # Calculate axis limits and color limits for this row
        all_row_points = [points]  # Original points

        # All patches (subsampled)
        for j in range(patches.shape[0]):
            patch_points = patches[j] + centers[j]
            all_row_points.append(patch_points)

        # Reconstructed patches
        for j in range(x_hat.shape[0]):
            patch_points = x_hat[j] + masked_centers[j]
            all_row_points.append(patch_points)

        # Calculate limits
        xlim, ylim, zlim = get_axis_limits(all_row_points)
        vmin, vmax = get_color_limits(all_row_points)

        # Get the leftmost axis of this row
        ax1 = fig.add_subplot(n_rows, n_cols, i * n_cols + 1, projection="3d")

        # Add row label using set_ylabel
        if publication_mode:
            label_text = plot_name
        else:
            label_text = f"{plot_name} (idx: {idx})"

        ax1.set_ylabel(label_text, fontsize=10, fontweight="bold", rotation=90)
        ax1.yaxis.set_label_coords(-0.3, 0.5)

        # Column 1: Original
        plot_original_voxel(ax1, points, title="", vmin=vmin, vmax=vmax)
        apply_axis_limits(ax1, xlim, ylim, zlim)

        # Column 2: Subsampled (all patches)
        ax2 = fig.add_subplot(n_rows, n_cols, i * n_cols + 2, projection="3d")
        plot_subsampled_patches(ax2, patches, centers, title="", vmin=vmin, vmax=vmax)
        apply_axis_limits(ax2, xlim, ylim, zlim)

        # Column 3: Masked (visible patches only)
        ax3 = fig.add_subplot(n_rows, n_cols, i * n_cols + 3, projection="3d")
        plot_visible_patches_only(
            ax3, patches, centers, mask, title="", vmin=vmin, vmax=vmax
        )
        apply_axis_limits(ax3, xlim, ylim, zlim)

        # Column 4: Reconstructed
        ax4 = fig.add_subplot(n_rows, n_cols, i * n_cols + 4, projection="3d")
        plot_full_reconstruction(
            ax4,
            patches,
            centers,
            mask,
            x_hat,
            masked_centers,
            title="",
            vmin=vmin,
            vmax=vmax,
        )
        apply_axis_limits(ax4, xlim, ylim, zlim)

    # Adjust layout - more left margin for row labels
    plt.subplots_adjust(
        left=0.15, right=0.98, top=0.94, bottom=0.02, wspace=0.05, hspace=0.1
    )

    # Save the figure
    output_path = os.path.join(output_dir, "reconstruction_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved to: {output_path}")

    # Also save as PDF
    pdf_path = os.path.join(output_dir, "reconstruction_grid.pdf")
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Recreate the same visualization for PDF
    # Add column titles - properly aligned
    left_margin = 0.15
    right_margin = 0.98
    plot_width = right_margin - left_margin
    for j, title in enumerate(col_titles):
        col_center = left_margin + (j + 0.5) * plot_width / n_cols
        fig.text(
            col_center,
            0.98,
            title,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    # Process each voxel again for PDF
    for i, voxel_data in enumerate(all_voxel_data):
        # Extract data
        plot_name = voxel_data["plot_name"]
        idx = voxel_data["idx"]
        points = voxel_data["points"]
        patches = voxel_data["patches"]
        centers = voxel_data["centers"]
        mask = voxel_data["mask"]
        x_hat = voxel_data["x_hat"]
        masked_centers = voxel_data["masked_centers"]
        voxel_data["num_points"]
        voxel_data["num_masked"]

        # Recalculate limits (same as before)
        all_row_points = [points]
        for j in range(patches.shape[0]):
            patch_points = patches[j] + centers[j]
            all_row_points.append(patch_points)
        for j in range(x_hat.shape[0]):
            patch_points = x_hat[j] + masked_centers[j]
            all_row_points.append(patch_points)

        xlim, ylim, zlim = get_axis_limits(all_row_points)
        vmin, vmax = get_color_limits(all_row_points)

        # Get the leftmost axis of this row
        ax1 = fig.add_subplot(n_rows, n_cols, i * n_cols + 1, projection="3d")

        # Add row label using set_ylabel
        if publication_mode:
            label_text = plot_name
        else:
            label_text = f"{plot_name} (idx: {idx})"

        ax1.set_ylabel(label_text, fontsize=10, fontweight="bold", rotation=90)
        ax1.yaxis.set_label_coords(-0.3, 0.5)

        # Create all 4 subplots for this row
        plot_original_voxel(ax1, points, title="", vmin=vmin, vmax=vmax)
        apply_axis_limits(ax1, xlim, ylim, zlim)

        ax2 = fig.add_subplot(n_rows, n_cols, i * n_cols + 2, projection="3d")
        plot_subsampled_patches(ax2, patches, centers, title="", vmin=vmin, vmax=vmax)
        apply_axis_limits(ax2, xlim, ylim, zlim)

        ax3 = fig.add_subplot(n_rows, n_cols, i * n_cols + 3, projection="3d")
        plot_visible_patches_only(
            ax3, patches, centers, mask, title="", vmin=vmin, vmax=vmax
        )
        apply_axis_limits(ax3, xlim, ylim, zlim)

        ax4 = fig.add_subplot(n_rows, n_cols, i * n_cols + 4, projection="3d")
        plot_full_reconstruction(
            ax4,
            patches,
            centers,
            mask,
            x_hat,
            masked_centers,
            title="",
            vmin=vmin,
            vmax=vmax,
        )
        apply_axis_limits(ax4, xlim, ylim, zlim)

    plt.subplots_adjust(
        left=0.15, right=0.98, top=0.94, bottom=0.02, wspace=0.05, hspace=0.1
    )
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"PDF saved to: {pdf_path}")


def visualize_labeled_reconstructions(
    checkpoint_path="/home/mja78/work/tlsPT/checkpoints_saved/vits_radius0.2_neighbors32_mr07_uldata1.0_best_loss0.0018.ckpt",
    config_path="../configs/pretrain/pretrain_vits_mr07.yaml",
    output_dir="saved_plots",
    num_samples=6,
    max_points=33000,
    specific_indices=None,  # List of specific voxel indices to visualize
    publication_mode=False,  # Whether to hide index values
):
    """Visualize reconstructions with leaf/wood labels

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config
        output_dir: Directory to save outputs
        num_samples: Number of samples to visualize (if not using specific_indices)
        max_points: Maximum points per voxel
        specific_indices: List of specific voxel indices to visualize
        publication_mode: If True, hide index values in labels for cleaner publication figures
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    print(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)

    # Manually resolve config references
    embedding_dim = config.model.embedding_dim
    config.model.transencoder_config.embed_dim = embedding_dim
    config.model.transdecoder_config.embed_dim = embedding_dim
    config.model.total_epochs = config.max_epochs

    # Initialize model from config using Hydra utils
    print("Initializing model from config")
    model = hydra.utils.instantiate(config.model)

    # Load checkpoint weights
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load model weights
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print("Model loaded successfully")

    # Try labeled dataset paths
    labeled_split_files = [
        "../data/tlspt_labelled/plot_octrees/hjfo-finl/hjfo-finl-splits.csv",
        "../data/tlspt_labelled/plot_octrees/hjfo-poll/hjfo-poll-splits.csv",
        "../data/tlspt_labelled/plot_octrees/hjfo-spal/hjfo-spal-splits.csv",
    ]

    # Find available split files
    available_splits = []
    for split_file in labeled_split_files:
        if os.path.exists(split_file):
            available_splits.append(split_file)
            print(f"Found labeled split file: {split_file}")
            # Don't break - we want ALL split files!

    if available_splits:
        # Use merged labeled dataset
        print(
            f"\nLoading merged labeled dataset from {len(available_splits)} split files"
        )

        dataset = MergedOctreeDataset(
            split_files=available_splits,
            split="val",  # Use validation split
            scales=2,  # Same scale for all datasets
            feature_names=["scalar_truth"],  # Load labels
            normalize=False,  # Don't normalize for visualization
            transform=None,
            min_points=512,
        )

        print(f"Dataset size: {len(dataset)} voxels")

        # Filter by point count
        if specific_indices:
            # Use specific indices if provided
            indices = specific_indices
            print(f"\nUsing specific labeled voxel indices: {indices}")

            # Verify these indices are valid and meet criteria
            valid_indices = []
            for idx in indices:
                if idx >= len(dataset):
                    print(
                        f"Warning: Index {idx} is out of range (dataset size: {len(dataset)})"
                    )
                    continue
                try:
                    data = dataset[idx]
                    points = data["points"]
                    if points.shape[0] > max_points:
                        print(
                            f"Warning: Voxel {idx} has {points.shape[0]} points (> {max_points}), skipping"
                        )
                        continue
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Warning: Could not load voxel {idx}: {e}")
                    continue

            indices = valid_indices
            if len(indices) == 0:
                print("No valid indices found!")
                return
        else:
            # Filter and randomly sample
            valid_indices = []
            for idx in range(len(dataset)):
                try:
                    data = dataset[idx]
                    points = data["points"]
                    if points.shape[0] <= max_points:
                        valid_indices.append(idx)
                except:
                    continue

            print(
                f"Found {len(valid_indices)} labeled voxels with <= {max_points} points"
            )

            if len(valid_indices) == 0:
                print(f"No labeled voxels found with <= {max_points} points!")
                return

            # Visualize with labels
            num_to_plot = min(num_samples, len(valid_indices))
            indices = np.random.choice(valid_indices, num_to_plot, replace=False)
            print(f"Randomly selected labeled indices: {sorted(indices.tolist())}")

        # Process all voxels to collect data
        all_voxel_data = []

        for i, idx in enumerate(indices):
            # Get plot name first
            dataset_idx, local_idx = dataset.idxs[idx]
            hdf5_path, _, _ = dataset.datasets[dataset_idx].voxels_to_load[local_idx]
            plot_name = os.path.basename(os.path.dirname(hdf5_path))
            plot_name = clean_plot_name(plot_name)  # Clean the plot name
            print(
                f"\nProcessing labeled voxel {i+1}/{len(indices)} - Index: {idx} from plot {plot_name}"
            )

            try:
                # Load voxel data with labels
                data = dataset[idx]
                points = data["points"]
                labels = data["features"].squeeze(-1)  # Remove feature dimension
                lengths = data["lengths"]

                # Print number of points in original voxel
                print(f"Original voxel has {points.shape[0]} points")

                # Double-check point count
                if points.shape[0] > max_points:
                    print(
                        f"Skipping voxel with {points.shape[0]} points (> {max_points})"
                    )
                    continue

                # Prepare input for model
                batch = {
                    "points": points.unsqueeze(0),
                    "lengths": torch.tensor([lengths]),
                }

                with torch.no_grad():
                    # Get patches and centers
                    patches, centers = model.group(batch["points"], batch["lengths"])

                    # Get mask
                    mask = model.mask_generator(centers)

                    # Run forward pass components
                    get_masked(patches, mask)
                    masked_centers = get_masked(centers, mask)

                    x_vis, _, vis_pos_embeddings = model.forward_encoder(
                        patches, centers
                    )

                    masked_pos_embeddings = model.pos_encoder(masked_centers)
                    B, N, _ = masked_pos_embeddings.shape
                    mask_tokens = model.mask_token.expand(B, N, -1)

                    x_full = torch.cat((x_vis, mask_tokens), dim=1)
                    full_pos_embeddings = torch.cat(
                        (vis_pos_embeddings, masked_pos_embeddings), dim=1
                    )

                    if model.encoder_to_decoder_proj is not None:
                        x_full = model.encoder_to_decoder_proj(x_full)
                        full_pos_embeddings = model.encoder_to_decoder_proj(
                            full_pos_embeddings
                        )

                    x_hat = model.forward_decoder(x_full, full_pos_embeddings, N)

                # Store all the data for this voxel
                voxel_data = {
                    "plot_name": plot_name,
                    "idx": idx,
                    "points": points.cpu().numpy(),
                    "labels": labels.cpu().numpy(),
                    "patches": patches.squeeze(0).cpu().numpy(),
                    "centers": centers.squeeze(0).cpu().numpy(),
                    "mask": mask.squeeze(0).cpu().numpy(),
                    "x_hat": x_hat.squeeze(0).cpu().numpy(),
                    "masked_centers": masked_centers.squeeze(0).cpu().numpy(),
                    "num_points": points.shape[0],
                    "num_masked": mask.sum().item(),
                }

                all_voxel_data.append(voxel_data)

            except Exception as e:
                print(f"Error processing labeled voxel from {plot_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Create grid visualization for labeled data
        n_rows = len(all_voxel_data)
        n_cols = 4  # Original, Subsampled, Masked, Reconstructed

        # Create figure with appropriate size
        fig_width = 4.5 * n_cols  # Slightly wider for better spacing
        fig_height = 4 * n_rows + 0.5  # Less extra space needed for titles
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Add column titles - properly aligned
        col_titles = ["Original", "Subsampled", "Masked", "Reconstructed"]
        left_margin = 0.15
        right_margin = 0.98
        plot_width = right_margin - left_margin
        for j, title in enumerate(col_titles):
            col_center = left_margin + (j + 0.5) * plot_width / n_cols
            fig.text(
                col_center,
                0.98,
                title,
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
            )

        # Process each voxel
        for i, voxel_data in enumerate(all_voxel_data):
            # Extract data
            plot_name = voxel_data["plot_name"]
            idx = voxel_data["idx"]
            points = voxel_data["points"]
            labels = voxel_data["labels"]
            patches = voxel_data["patches"]
            centers = voxel_data["centers"]
            mask = voxel_data["mask"]
            x_hat = voxel_data["x_hat"]
            masked_centers = voxel_data["masked_centers"]
            voxel_data["num_points"]
            voxel_data["num_masked"]

            # Calculate axis limits for this row
            all_row_points = [points]  # Original points

            # All patches
            for j in range(patches.shape[0]):
                patch_points = patches[j] + centers[j]
                all_row_points.append(patch_points)

            # Reconstructed patches
            for j in range(x_hat.shape[0]):
                patch_points = x_hat[j] + masked_centers[j]
                all_row_points.append(patch_points)

            # Calculate limits
            xlim, ylim, zlim = get_axis_limits(all_row_points)

            # Get the leftmost axis of this row
            ax1 = fig.add_subplot(n_rows, n_cols, i * n_cols + 1, projection="3d")

            # Add row label using set_ylabel
            if publication_mode:
                label_text = plot_name
            else:
                label_text = f"{plot_name} (idx: {idx})"

            ax1.set_ylabel(label_text, fontsize=10, fontweight="bold", rotation=90)
            ax1.yaxis.set_label_coords(-0.3, 0.5)

            # Column 1: Original labeled
            plot_labeled_original(ax1, points, labels, title="")
            apply_axis_limits(ax1, xlim, ylim, zlim)

            # Column 2: Subsampled with labels
            ax2 = fig.add_subplot(n_rows, n_cols, i * n_cols + 2, projection="3d")
            plot_labeled_subsampled(ax2, patches, centers, points, labels, title="")
            apply_axis_limits(ax2, xlim, ylim, zlim)

            # Column 3: Masked (visible patches only)
            ax3 = fig.add_subplot(n_rows, n_cols, i * n_cols + 3, projection="3d")
            plot_labeled_visible_only(
                ax3, patches, centers, mask, points, labels, title=""
            )
            apply_axis_limits(ax3, xlim, ylim, zlim)

            # Column 4: Reconstructed
            ax4 = fig.add_subplot(n_rows, n_cols, i * n_cols + 4, projection="3d")
            plot_labeled_reconstruction(
                ax4,
                patches,
                centers,
                mask,
                x_hat,
                masked_centers,
                points,
                labels,
                title="",
            )
            apply_axis_limits(ax4, xlim, ylim, zlim)

            # Add legend to first row only
            if i == 0:
                legend_elements = [
                    Patch(facecolor="dimgrey", label="Leaf"),
                    Patch(facecolor="red", label="Wood"),
                    Patch(facecolor="blue", label="Reconstructed"),
                ]
                ax4.legend(handles=legend_elements, loc="upper right", fontsize=8)

        # Adjust layout - more left margin for row labels
        plt.subplots_adjust(
            left=0.15, right=0.98, top=0.94, bottom=0.02, wspace=0.05, hspace=0.1
        )

        # Save the figure
        output_path = os.path.join(output_dir, "labeled_reconstruction_grid.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nLabeled visualization saved to: {output_path}")

        # Also save as PDF
        pdf_path = os.path.join(output_dir, "labeled_reconstruction_grid.pdf")
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Recreate for PDF (similar to above)
        left_margin = 0.15
        right_margin = 0.98
        plot_width = right_margin - left_margin
        for j, title in enumerate(col_titles):
            col_center = left_margin + (j + 0.5) * plot_width / n_cols
            fig.text(
                col_center,
                0.98,
                title,
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
            )

        for i, voxel_data in enumerate(all_voxel_data):
            plot_name = voxel_data["plot_name"]
            idx = voxel_data["idx"]
            points = voxel_data["points"]
            labels = voxel_data["labels"]
            patches = voxel_data["patches"]
            centers = voxel_data["centers"]
            mask = voxel_data["mask"]
            x_hat = voxel_data["x_hat"]
            masked_centers = voxel_data["masked_centers"]
            voxel_data["num_points"]
            voxel_data["num_masked"]

            # Recalculate limits
            all_row_points = [points]
            for j in range(patches.shape[0]):
                patch_points = patches[j] + centers[j]
                all_row_points.append(patch_points)
            for j in range(x_hat.shape[0]):
                patch_points = x_hat[j] + masked_centers[j]
                all_row_points.append(patch_points)

            xlim, ylim, zlim = get_axis_limits(all_row_points)

            # Get the leftmost axis of this row
            ax1 = fig.add_subplot(n_rows, n_cols, i * n_cols + 1, projection="3d")

            # Add row label using set_ylabel
            if publication_mode:
                label_text = plot_name
            else:
                label_text = f"{plot_name} (idx: {idx})"

            ax1.set_ylabel(label_text, fontsize=10, fontweight="bold", rotation=90)
            ax1.yaxis.set_label_coords(-0.3, 0.5)

            # Create all 4 subplots
            plot_labeled_original(ax1, points, labels, title="")
            apply_axis_limits(ax1, xlim, ylim, zlim)

            ax2 = fig.add_subplot(n_rows, n_cols, i * n_cols + 2, projection="3d")
            plot_labeled_subsampled(ax2, patches, centers, points, labels, title="")
            apply_axis_limits(ax2, xlim, ylim, zlim)

            ax3 = fig.add_subplot(n_rows, n_cols, i * n_cols + 3, projection="3d")
            plot_labeled_visible_only(
                ax3, patches, centers, mask, points, labels, title=""
            )
            apply_axis_limits(ax3, xlim, ylim, zlim)

            ax4 = fig.add_subplot(n_rows, n_cols, i * n_cols + 4, projection="3d")
            plot_labeled_reconstruction(
                ax4,
                patches,
                centers,
                mask,
                x_hat,
                masked_centers,
                points,
                labels,
                title="",
            )
            apply_axis_limits(ax4, xlim, ylim, zlim)

            if i == 0:
                legend_elements = [
                    Patch(facecolor="dimgrey", label="Leaf"),
                    Patch(facecolor="red", label="Wood"),
                    Patch(facecolor="blue", label="Reconstructed"),
                ]
                ax4.legend(handles=legend_elements, loc="upper right", fontsize=8)

        plt.subplots_adjust(
            left=0.15, right=0.98, top=0.94, bottom=0.02, wspace=0.05, hspace=0.1
        )
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"PDF saved to: {pdf_path}")
    else:
        print(
            "No labeled dataset found, using unlabeled reconstruction visualization only"
        )


if __name__ == "__main__":
    print("Starting reconstruction visualization...")

    # Optional: Check for command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Visualize TLS voxel reconstructions")
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Specific voxel indices to visualize (e.g., --indices 15 42 108)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of samples to visualize if not using --indices (default: 6)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=33000,
        help="Maximum points per voxel (default: 33000)",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Publication mode - hide index values in labels",
    )
    args = parser.parse_args()

    # Config path - adjust if needed
    config_path = "../configs/pretrain/pretrain_vits_mr07.yaml"

    # Try alternative paths if the default doesn't exist
    if not os.path.exists(config_path):
        alt_paths = [
            "configs/pretrain/pretrain_vits_mr07.yaml",
            "/home/mja78/work/tlsPT/configs/pretrain/pretrain_vits_mr07.yaml",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "configs/pretrain/pretrain_vits_mr07.yaml",
            ),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                config_path = alt_path
                print(f"Using config path: {config_path}")
                break
        else:
            print(
                f"Warning: Config file not found at {config_path} or alternative paths"
            )

    # Run unlabeled visualization with max_points filter
    visualize_reconstructions(
        config_path=config_path,
        output_dir="saved_plots",
        num_samples=args.num_samples,
        max_points=args.max_points,
        specific_indices=args.indices,
        publication_mode=args.publication,
    )

    # Try labeled visualization if data is available
    # visualize_labeled_reconstructions(
    #     config_path=config_path,
    #     output_dir="saved_plots",
    #     num_samples=args.num_samples,
    #     max_points=args.max_points,
    #     specific_indices=args.indices,
    #     publication_mode=args.publication
    # )

    print("\nVisualization complete!")
