from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb

warnings.filterwarnings("ignore")

# Initialize wandb
api = wandb.Api()

# Constants
PROJECT = "mja2106/FINAL_NOWEIGHT_TUNE_TLSPT_2025"
OUTPUT_DIR = Path("saved_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define metrics to plot
METRICS = {
    "bal_acc": "Balanced Accuracy",
    "miou": "Mean IoU",
}

# Which checkpoint to evaluate
CHECKPOINT = "last"  # or "best", "first", "step97"

# Freeze types to plot
FREEZE_TYPES = {
    "full": "Full Finetune",
    "frozen": "Frozen Encoder",
    "scratch": "From Scratch",
    "scheduled": "Scheduled Unfreeze",
}

# Color scheme for freeze types - using matplotlib's default shades
FREEZE_COLORS = {
    "scratch": "gray",  # matplotlib's default gray
    "full": "tab:blue",  # matplotlib's default blue
    "frozen": "tab:orange",  # matplotlib's default orange
    "scheduled": "tab:green",  # matplotlib's default green
}

# Architecture mapping
ARCH_MAPPING = {
    "vits": "ViT-S",
    "vitb": "ViT-B",
    "vitl": "ViT-L",
}


def setup_matplotlib():
    """
    Configure matplotlib for publication-quality figures.
    Remove font-specific settings to avoid errors.
    """
    mpl.rcParams["axes.labelsize"] = 14  # Increased from 12
    mpl.rcParams["axes.titlesize"] = 16  # Increased from 14
    mpl.rcParams["xtick.labelsize"] = 14  # Increased from 10
    mpl.rcParams["ytick.labelsize"] = 14  # Increased from 10
    mpl.rcParams["legend.fontsize"] = 14  # Increased from 10
    mpl.rcParams["figure.titlesize"] = 16  # Kept the same
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.1


def parse_checkpoint_name(checkpoint_name):
    """Extract architecture and unlabeled data percentage from checkpoint name"""
    if checkpoint_name == "scratch":
        return "scratch", 0.0

    # Extract architecture (vits, vitb, vitl)
    arch_match = re.match(r"(vit[sbl])", checkpoint_name.lower())
    arch = arch_match.group(1) if arch_match else "unknown"

    # Extract unlabeled data percentage
    uldata_match = re.search(r"uldata([\d.]+)", checkpoint_name)
    uldata_pct = float(uldata_match.group(1)) * 100 if uldata_match else None

    return arch, uldata_pct


def fetch_runs():
    """Fetch all runs from the W&B project"""
    print(f"Fetching runs from {PROJECT}...")
    runs = api.runs(PROJECT)

    data = []
    missing_flops_count = 0
    filtered_count = 0

    for run in runs:
        # Filter out unfinished or crashed runs
        if run.state != "finished":
            filtered_count += 1
            continue

        # Get config and summary data
        config = run.config
        summary = run.summary._json_dict

        # Skip if no ablation data
        if "ablation/freeze_type" not in config:
            continue

        # Extract checkpoint name and freeze type
        checkpoint = config.get("ablation/checkpoint", "scratch")
        freeze_type = config.get("ablation/freeze_type")

        # Parse checkpoint info
        arch, uldata_pct = parse_checkpoint_name(checkpoint)

        # Skip if not 100% unlabeled data (except scratch)
        if checkpoint != "scratch" and uldata_pct != 100.0:
            continue

        # Check if we have flops/total_training
        if "flops/total_training" not in summary:
            missing_flops_count += 1
            continue

        # Extract data
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "freeze_type": freeze_type,
            "train_pct": config.get(
                "ablation/train_pct", config.get("ablation/train_percent")
            ),
            "checkpoint": checkpoint,
            "arch": arch,
            "uldata_pct": uldata_pct,
            "train_flops": summary["flops/total_training"],
            "run_index": config.get("ablation/run_index", 0),
        }

        # Extract metrics
        for metric in METRICS:
            metric_key = f"test/{metric}_epoch_{CHECKPOINT}"
            alternative_keys = [
                f"test/{metric}_{CHECKPOINT}",
                f"test/{metric}_{CHECKPOINT}_epoch",
            ]

            if metric_key in summary:
                row[f"{metric}_{CHECKPOINT}"] = summary[metric_key]
            else:
                for key in alternative_keys:
                    if key in summary:
                        row[f"{metric}_{CHECKPOINT}"] = summary[key]
                        break

        data.append(row)

    df = pd.DataFrame(data)

    print(f"\nTotal runs with 100% unlabeled data (or scratch): {len(df)}")
    print(f"Runs filtered for not being finished: {filtered_count}")
    print(f"Runs missing flops/total_training: {missing_flops_count}")

    # Filter out rows with missing essential data
    df = df.dropna(subset=["freeze_type", "train_pct", "train_flops"])
    print(f"After filtering: {len(df)} runs")

    # Show distribution of freeze types
    print("\nFreeze type distribution:")
    for ft, count in df["freeze_type"].value_counts().items():
        print(f"  {ft}: {count} runs")

    return df


def aggregate_runs(df):
    """Aggregate multiple runs with same parameters, computing mean values"""
    group_cols = ["freeze_type", "train_pct", "checkpoint", "arch", "uldata_pct"]

    # For aggregation, we want to average both metrics and FLOPs
    metric_cols = [f"{m}_{CHECKPOINT}" for m in METRICS]
    existing_metric_cols = [col for col in metric_cols if col in df.columns]

    agg_dict = {col: "mean" for col in existing_metric_cols}
    agg_dict["train_flops"] = "mean"
    agg_dict["run_id"] = "count"  # Count number of runs

    df_agg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    df_agg.rename(columns={"run_id": "num_runs"}, inplace=True)

    print(f"\nAfter aggregation: {len(df_agg)} unique configurations")

    # Check for configurations with different numbers of runs
    run_counts = df_agg["num_runs"].value_counts().sort_index()
    print("\nRun count distribution:")
    for count, freq in run_counts.items():
        print(f"  {count} runs: {freq} configurations")

    return df_agg


def compute_efficient_frontier_proper(freeze_data_dict, metric_col):
    """
    Compute the efficient frontier by finding where curves intersect and which
    segments dominate.

    Args:
        freeze_data_dict: Dictionary mapping freeze_type to dataframe with 'train_flops' and metric_col
        metric_col: Name of the metric column

    Returns:
        frontier_flops, frontier_metrics: Arrays of points on the frontier
    """
    # Build list of all segments from all curves
    segments = []

    for freeze_type, data in freeze_data_dict.items():
        if len(data) < 2:  # Need at least 2 points for a segment
            continue

        # Sort by FLOPs
        sorted_data = data.sort_values("train_flops")
        x = sorted_data["train_flops"].values
        y = sorted_data[metric_col].values

        # Add each segment
        for i in range(len(x) - 1):
            segments.append(
                {
                    "x1": x[i],
                    "y1": y[i],
                    "x2": x[i + 1],
                    "y2": y[i + 1],
                    "freeze_type": freeze_type,
                }
            )

    if len(segments) == 0:
        return np.array([]), np.array([])

    # Find all x-coordinates where we need to check (segment endpoints and intersections)
    x_coords = set()

    # Add all segment endpoints
    for seg in segments:
        x_coords.add(seg["x1"])
        x_coords.add(seg["x2"])

    # Find intersection points
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments[i + 1 :], i + 1):
            # Check if x-ranges overlap
            x_min = max(seg1["x1"], seg2["x1"])
            x_max = min(seg1["x2"], seg2["x2"])

            if x_min < x_max:  # Segments might intersect
                # Calculate intersection
                # Line 1: y = m1*x + b1
                # Line 2: y = m2*x + b2
                dx1 = seg1["x2"] - seg1["x1"]
                dy1 = seg1["y2"] - seg1["y1"]
                dx2 = seg2["x2"] - seg2["x1"]
                dy2 = seg2["y2"] - seg2["y1"]

                if dx1 != 0 and dx2 != 0:
                    m1 = dy1 / dx1
                    m2 = dy2 / dx2
                    b1 = seg1["y1"] - m1 * seg1["x1"]
                    b2 = seg2["y1"] - m2 * seg2["x1"]

                    if abs(m1 - m2) > 1e-10:  # Not parallel
                        x_int = (b2 - b1) / (m1 - m2)
                        if x_min <= x_int <= x_max:
                            x_coords.add(x_int)

    # Sort x-coordinates
    x_sorted = sorted(x_coords)

    # For each x-coordinate, find which segment gives maximum y
    frontier_points = []

    for x in x_sorted:
        max_y = -np.inf

        # Check all segments that contain this x
        for seg in segments:
            if seg["x1"] <= x <= seg["x2"]:
                # Interpolate y at this x
                if seg["x2"] != seg["x1"]:
                    t = (x - seg["x1"]) / (seg["x2"] - seg["x1"])
                    y = seg["y1"] + t * (seg["y2"] - seg["y1"])
                else:
                    y = seg["y1"]

                if y > max_y:
                    max_y = y

        if max_y > -np.inf:
            # Only add if it improves the frontier
            if len(frontier_points) == 0 or max_y > frontier_points[-1][1]:
                frontier_points.append((x, max_y))

    if len(frontier_points) == 0:
        return np.array([]), np.array([])

    frontier_x = np.array([p[0] for p in frontier_points])
    frontier_y = np.array([p[1] for p in frontier_points])

    return frontier_x, frontier_y


def create_pareto_plot(df, metric, architecture, publication=False):
    """Create a pareto plot with training FLOPs on x-axis and efficient frontier"""

    # Filter for specific architecture
    arch_data = df[
        (df["arch"] == architecture)
        | (df["checkpoint"] == "scratch")  # Include scratch as baseline
    ].copy()

    metric_col = f"{metric}_{CHECKPOINT}"

    # Drop rows with missing metric values
    arch_data = arch_data.dropna(subset=[metric_col, "train_flops"])

    if len(arch_data) == 0:
        print(f"No data for {metric} - {architecture}")
        return None

    # Create figure with narrower aspect ratio for publication
    figsize = (6, 8) if publication else (12, 8)
    fig, ax = plt.subplots(figsize=figsize)

    # Collect data by freeze type for frontier calculation
    freeze_data_dict = {}

    # Plot each freeze type
    for freeze_type in [
        "scratch",
        "full",
        "frozen",
        "scheduled",
    ]:  # Order matters for legend
        if freeze_type not in FREEZE_TYPES:
            continue

        data = arch_data[arch_data["freeze_type"] == freeze_type]
        if len(data) == 0:
            continue

        # Sort by FLOPs
        data = data.sort_values("train_flops")

        # Store data for frontier calculation
        freeze_data_dict[freeze_type] = data

        # Get color for this freeze type
        color = FREEZE_COLORS.get(freeze_type, "black")

        # Plot each freeze type
        line = ax.plot(
            data["train_flops"],
            data[metric_col],
            marker="o",
            label=FREEZE_TYPES[freeze_type],
            linewidth=2.5,
            markersize=10,
            alpha=0.8,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )[0]

        # Add text labels with training data percentage (only for scheduled and scratch)
        if freeze_type in ["scheduled", "scratch"]:
            for i, (_, row) in enumerate(data.iterrows()):
                train_pct_label = f"{row['train_pct']*100:.0f}%"

                # Different offsets for different freeze types
                if freeze_type == "scheduled":
                    offset = (10, -15)  # bottom right
                elif freeze_type == "scratch":
                    offset = (-10, 10)  # top left
                else:
                    offset = (10, 5)  # default

                # Offset text to avoid overlap with point
                ax.annotate(
                    train_pct_label,
                    xy=(row["train_flops"], row[metric_col]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=12,
                    color=color,
                    weight="bold",
                    ha="left" if offset[0] > 0 else "right",
                    va="bottom" if offset[1] > 0 else "top",
                )

    # Compute and plot efficient frontier using proper method
    frontier_flops, frontier_metrics = compute_efficient_frontier_proper(
        freeze_data_dict, metric_col
    )

    if len(frontier_flops) > 0:
        # Plot efficient frontier
        ax.plot(
            frontier_flops,
            frontier_metrics,
            "k--",
            linewidth=3,
            label="Efficient Frontier",
            alpha=0.7,
            zorder=10,
        )

    # Formatting
    ax.set_xlabel("Training FLOPs", fontsize=14)
    ax.set_ylabel(METRICS[metric], fontsize=14)

    # Set log scale for x-axis
    ax.set_xscale("log")

    # Title
    ax.set_title(
        f"{METRICS[metric]} vs Training FLOPs - Full Finetune",
        fontsize=16,
        fontweight="bold",
    )

    # Grid
    ax.grid(True, alpha=0.3, which="both")

    # Legend - no shadow
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False)

    # Add minor gridlines
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()

    plt.tight_layout()

    return fig


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create Pareto plots from W&B runs")
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Create separate plots with narrower aspect ratio for publication",
    )
    args = parser.parse_args()

    # Set up matplotlib for publication quality
    setup_matplotlib()

    # Fetch and process data
    df = fetch_runs()

    if len(df) == 0:
        print("No runs found matching criteria")
        return

    # Aggregate duplicate runs
    df_agg = aggregate_runs(df)

    # Save raw data
    df_agg.to_csv(OUTPUT_DIR / "pareto_plot_data_averaged.csv", index=False)
    print(f"\nSaved raw data to {OUTPUT_DIR / 'pareto_plot_data_averaged.csv'}")

    # Get unique architectures (excluding scratch)
    architectures = sorted(df_agg[df_agg["arch"] != "scratch"]["arch"].unique())
    print(f"\nArchitectures found: {architectures}")

    # Print summary statistics
    print("\n=== Training FLOPs Summary ===")
    print(
        "(Showing both raw values and log10 for easier interpretation with log scale plots)"
    )
    for freeze_type in FREEZE_TYPES:
        ft_data = df_agg[df_agg["freeze_type"] == freeze_type]
        if len(ft_data) > 0:
            print(f"\n{FREEZE_TYPES[freeze_type]}:")
            print(f"  Mean training FLOPs: {ft_data['train_flops'].mean():.2e}")
            print(
                f"  Range: {ft_data['train_flops'].min():.2e} - {ft_data['train_flops'].max():.2e}"
            )
            print(
                f"  Log10 range: {np.log10(ft_data['train_flops'].min()):.1f} - {np.log10(ft_data['train_flops'].max()):.1f}"
            )

    # Create plots
    if args.publication:
        # Create separate plots for publication
        for metric in METRICS:
            for arch in architectures:
                print(f"\nCreating pareto plot for {metric} - {arch}")
                fig = create_pareto_plot(df_agg, metric, arch, publication=True)
                if fig:
                    # Save as individual file
                    filename = f"pareto_{metric}_{arch}.pdf"
                    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", dpi=300)
                    print(f"Saved plot to {OUTPUT_DIR / filename}")
                    plt.close(fig)
    else:
        # Original behavior - combine all plots in one PDF
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(OUTPUT_DIR / "pareto_plots_averaged.pdf") as pdf:
            for metric in METRICS:
                # Individual plots for each architecture
                for arch in architectures:
                    print(f"\nCreating pareto plot for {metric} - {arch}")
                    fig = create_pareto_plot(df_agg, metric, arch, publication=False)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)

        print(f"\nSaved plots to {OUTPUT_DIR / 'pareto_plots_averaged.pdf'}")


if __name__ == "__main__":
    main()
