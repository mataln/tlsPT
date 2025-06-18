#!/usr/bin/env python3
"""
Script to visualize voxelized TLS data with reconstruction from trained PointMAE model

Creates a 3-column visualization:
1. Left: Original voxel data (gradient colored by height for unlabeled, leaf/wood colors for labeled)
2. Center: Voxel with masked patches completely removed (showing only visible patches)
3. Right: Full reconstruction (visible patches + reconstructed masked patches)

For labeled data:
- Leaves (label 0) = dark grey
- Wood (label 1) = red
- Reconstructed patches = blue

Uses validation split for better coverage of the data.
All axes in each row share the same limits for better comparison.
Prints the number of points in each original voxel.

Uses the model config file to instantiate the model (same approach as train.py),
then loads the checkpoint weights.

Save as: plots/reconstruction_grid.py
"""

from __future__ import annotations

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

from tlspt.datamodules.components.octree_dataset import OctreeDataset
from tlspt.models.utils import get_masked


def setup_3d_plot(title="Voxel Visualization", figsize=(8, 6)):
    """Setup a 3D plot with good viewing angle"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=45)
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


def apply_axis_limits(ax, xlim, ylim, zlim):
    """Apply axis limits to a 3D plot"""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


def plot_original_voxel(ax, points, title=""):
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
    )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])


def plot_visible_patches_only(ax, patches, centers, mask, title=""):
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
        )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])


def plot_full_reconstruction(
    ax, patches, centers, mask, reconstructed_patches, masked_centers, title=""
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
        )

    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])


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

    # Add legend
    legend_elements = [
        Patch(facecolor="dimgrey", label="Leaf"),
        Patch(facecolor="red", label="Wood"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")


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
):
    """Load model and visualize reconstructions"""

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
    ]

    # Find available split file
    available_splits = []
    for split_file in split_files:
        if os.path.exists(split_file):
            available_splits.append(split_file)
            print(f"Found split file: {split_file}")
            break

    if not available_splits:
        print("No split files found. Please check your data directory.")
        return

    # Create dataset
    split_file = available_splits[0]
    print(f"\nLoading dataset from {split_file}")

    dataset = OctreeDataset(
        split_file=split_file,
        split="val",  # Use validation split
        scale=2,
        feature_names=None,  # No features needed for reconstruction
        normalize=True,
        transform=None,
        min_points=512,
    )

    print(f"Dataset size: {len(dataset)} voxels")

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    # Sample some voxels
    num_to_plot = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_to_plot, replace=False)

    # Create a multi-page PDF
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = os.path.join(output_dir, "reconstruction_visualizations.pdf")

    with PdfPages(pdf_path) as pdf:
        for i, idx in enumerate(indices):
            # Get plot name first (before try block)
            hdf5_path, _, _ = dataset.voxels_to_load[idx]
            plot_name = os.path.basename(os.path.dirname(hdf5_path))
            print(f"\nProcessing voxel {i+1}/{num_to_plot} from plot {plot_name}")

            try:
                # Load voxel data
                data = dataset[idx]
                points = data["points"]
                lengths = data["lengths"]

                # Print number of points in original voxel
                print(f"Original voxel has {points.shape[0]} points")

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
                    x_vis, _, vis_pos_embeddings = model.forward_encoder(
                        patches, centers
                    )

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

                # Calculate axis limits for all data in this row
                # Need to collect all points that will be displayed
                points_np = points.squeeze(0).cpu().numpy()
                patches_np = patches.squeeze(0).cpu().numpy()
                centers_np = centers.squeeze(0).cpu().numpy()
                mask_np = mask.squeeze(0).cpu().numpy()
                x_hat_np = x_hat.squeeze(0).cpu().numpy()
                masked_centers_np = masked_centers.squeeze(0).cpu().numpy()

                # Collect all points for limit calculation
                all_row_points = [points_np]  # Original points

                # Visible patches
                vis_patches = patches_np[~mask_np]
                vis_centers = centers_np[~mask_np]
                for i in range(vis_patches.shape[0]):
                    patch_points = vis_patches[i] + vis_centers[i]
                    all_row_points.append(patch_points)

                # Reconstructed patches
                for i in range(x_hat_np.shape[0]):
                    patch_points = x_hat_np[i] + masked_centers_np[i]
                    all_row_points.append(patch_points)

                # Calculate limits
                xlim, ylim, zlim = get_axis_limits(all_row_points)

                # Create visualization
                fig = plt.figure(figsize=(18, 6))

                # Left: Original voxel
                ax1 = fig.add_subplot(131, projection="3d")
                plot_original_voxel(
                    ax1, points.squeeze(0), title=f"Original Voxel\n{plot_name}"
                )
                apply_axis_limits(ax1, xlim, ylim, zlim)

                # Center: Visible patches only (masked patches removed)
                ax2 = fig.add_subplot(132, projection="3d")
                plot_visible_patches_only(
                    ax2,
                    patches.squeeze(0),
                    centers.squeeze(0),
                    mask.squeeze(0),
                    title=f"With {mask.sum().item()} Patches Removed\n{plot_name}",
                )
                apply_axis_limits(ax2, xlim, ylim, zlim)

                # Right: Full reconstruction
                ax3 = fig.add_subplot(133, projection="3d")
                plot_full_reconstruction(
                    ax3,
                    patches.squeeze(0),
                    centers.squeeze(0),
                    mask.squeeze(0),
                    x_hat.squeeze(0),
                    masked_centers.squeeze(0),
                    title=f"Full Reconstruction\n{plot_name}",
                )
                apply_axis_limits(ax3, xlim, ylim, zlim)

                # Save to PDF
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # Also save individual PNG for the first few
                if i < 3:
                    fig = plt.figure(figsize=(18, 6))
                    ax1 = fig.add_subplot(131, projection="3d")
                    plot_original_voxel(
                        ax1, points.squeeze(0), title=f"Original\n{plot_name}"
                    )
                    apply_axis_limits(ax1, xlim, ylim, zlim)

                    ax2 = fig.add_subplot(132, projection="3d")
                    plot_visible_patches_only(
                        ax2,
                        patches.squeeze(0),
                        centers.squeeze(0),
                        mask.squeeze(0),
                        title=f"Masked ({mask.sum().item()} removed)\n{plot_name}",
                    )
                    apply_axis_limits(ax2, xlim, ylim, zlim)

                    ax3 = fig.add_subplot(133, projection="3d")
                    plot_full_reconstruction(
                        ax3,
                        patches.squeeze(0),
                        centers.squeeze(0),
                        mask.squeeze(0),
                        x_hat.squeeze(0),
                        masked_centers.squeeze(0),
                        title=f"Reconstruction\n{plot_name}",
                    )
                    apply_axis_limits(ax3, xlim, ylim, zlim)

                    png_path = os.path.join(
                        output_dir, f"reconstruction_{plot_name}_{i}.png"
                    )
                    plt.savefig(png_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved individual plot: {png_path}")

            except Exception as e:
                print(f"Error processing voxel from {plot_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    print(f"\nVisualization saved to: {pdf_path}")


def visualize_labeled_reconstructions(
    checkpoint_path="/home/mja78/work/tlsPT/checkpoints_saved/vits_radius0.2_neighbors32_mr07_uldata1.0_best_loss0.0018.ckpt",
    config_path="../configs/pretrain/pretrain_vits_mr07.yaml",
    output_dir="saved_plots",
    num_samples=6,
):
    """Visualize reconstructions with leaf/wood labels"""

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

    # Find available split file
    available_splits = []
    for split_file in labeled_split_files:
        if os.path.exists(split_file):
            available_splits.append(split_file)
            print(f"Found labeled split file: {split_file}")
            break

    if available_splits:
        # Use labeled dataset
        split_file = available_splits[0]
        print(f"\nLoading labeled dataset from {split_file}")

        dataset = OctreeDataset(
            split_file=split_file,
            split="val",  # Use validation split
            scale=2,
            feature_names=["scalar_truth"],  # Load labels
            normalize=False,  # Don't normalize for visualization
            transform=None,
            min_points=512,
        )

        print(f"Dataset size: {len(dataset)} voxels")

        # Visualize with labels
        num_to_plot = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_to_plot, replace=False)

        pdf_path = os.path.join(output_dir, "labeled_reconstruction_visualizations.pdf")

        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(pdf_path) as pdf:
            for i, idx in enumerate(indices):
                # Get plot name first (before try block)
                hdf5_path, _, _ = dataset.voxels_to_load[idx]
                plot_name = os.path.basename(os.path.dirname(hdf5_path))
                print(
                    f"\nProcessing labeled voxel {i+1}/{num_to_plot} from plot {plot_name}"
                )

                try:
                    # Load voxel data with labels
                    data = dataset[idx]
                    points = data["points"]
                    labels = data["features"].squeeze(-1)  # Remove feature dimension
                    lengths = data["lengths"]

                    # Print number of points in original voxel
                    print(f"Original voxel has {points.shape[0]} points")

                    # Prepare input for model
                    batch = {
                        "points": points.unsqueeze(0),
                        "lengths": torch.tensor([lengths]),
                    }

                    with torch.no_grad():
                        # Get patches and centers
                        patches, centers = model.group(
                            batch["points"], batch["lengths"]
                        )

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

                    # Calculate axis limits for all data in this row
                    points_np = points.cpu().numpy()
                    patches_np = patches.squeeze(0).cpu().numpy()
                    centers_np = centers.squeeze(0).cpu().numpy()
                    mask_np = mask.squeeze(0).cpu().numpy()
                    x_hat_np = x_hat.squeeze(0).cpu().numpy()
                    masked_centers_np = masked_centers.squeeze(0).cpu().numpy()

                    # Collect all points for limit calculation
                    all_row_points = [points_np]  # Original points

                    # Visible patches
                    vis_patches = patches_np[~mask_np]
                    vis_centers = centers_np[~mask_np]
                    for i in range(vis_patches.shape[0]):
                        patch_points = vis_patches[i] + vis_centers[i]
                        all_row_points.append(patch_points)

                    # Reconstructed patches
                    for i in range(x_hat_np.shape[0]):
                        patch_points = x_hat_np[i] + masked_centers_np[i]
                        all_row_points.append(patch_points)

                    # Calculate limits
                    xlim, ylim, zlim = get_axis_limits(all_row_points)

                    # Create figure with 3 columns
                    fig = plt.figure(figsize=(18, 6))

                    # Plot 1: Original labeled points
                    ax1 = fig.add_subplot(131, projection="3d")
                    plot_labeled_original(
                        ax1,
                        points,
                        labels,
                        title=f"Original Labeled Points\n{plot_name}",
                    )
                    apply_axis_limits(ax1, xlim, ylim, zlim)

                    # Plot 2: Visible patches only
                    ax2 = fig.add_subplot(132, projection="3d")
                    plot_labeled_visible_only(
                        ax2,
                        patches.squeeze(0),
                        centers.squeeze(0),
                        mask.squeeze(0),
                        points,
                        labels,
                        title=f"With {mask.sum().item()} Patches Removed\n{plot_name}",
                    )
                    apply_axis_limits(ax2, xlim, ylim, zlim)

                    # Plot 3: Full reconstruction
                    ax3 = fig.add_subplot(133, projection="3d")
                    plot_labeled_reconstruction(
                        ax3,
                        patches.squeeze(0),
                        centers.squeeze(0),
                        mask.squeeze(0),
                        x_hat.squeeze(0),
                        masked_centers.squeeze(0),
                        points,
                        labels,
                        title=f"Full Reconstruction\n{plot_name}",
                    )
                    apply_axis_limits(ax3, xlim, ylim, zlim)

                    # Save to PDF
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                except Exception as e:
                    print(f"Error processing labeled voxel from {plot_name}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

        print(f"\nLabeled visualization saved to: {pdf_path}")
    else:
        print(
            "No labeled dataset found, using unlabeled reconstruction visualization only"
        )


if __name__ == "__main__":
    print("Starting reconstruction visualization...")

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

    # Run unlabeled visualization
    visualize_reconstructions(
        config_path=config_path, output_dir="saved_plots", num_samples=6
    )

    # Try labeled visualization if data is available
    visualize_labeled_reconstructions(
        config_path=config_path, output_dir="saved_plots", num_samples=6
    )

    print("\nVisualization complete!")
