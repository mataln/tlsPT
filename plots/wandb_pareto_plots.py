from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb

warnings.filterwarnings("ignore")

# Initialize wandb
api = wandb.Api()

# Constants
PROJECT = "mja2106/FINAL_NOWEIGHT_TUNE_TLSPT_2025"
OUTPUT_DIR = Path("saved_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

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

# Architecture mapping
ARCH_MAPPING = {
    "vits": "ViT-S",
    "vitb": "ViT-B",
    "vitl": "ViT-L",
}


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

    for run in runs:
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
    """Aggregate multiple runs with same parameters"""
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

    return df_agg


def compute_efficient_frontier(flops, metric_values):
    """
    Compute the efficient frontier for a set of points.
    Returns indices of points on the frontier.
    """
    # Create array of indices and sort by flops
    points = np.column_stack([flops, metric_values])
    indices = np.arange(len(flops))
    sorted_indices = indices[np.argsort(flops)]

    frontier_indices = []
    max_metric = -np.inf

    for idx in sorted_indices:
        if points[idx, 1] > max_metric:
            frontier_indices.append(idx)
            max_metric = points[idx, 1]

    return np.array(frontier_indices)


def create_pareto_plot(df, metric, architecture):
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

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect all points for efficient frontier calculation
    all_flops = []
    all_metrics = []
    all_labels = []

    # Plot each freeze type
    for freeze_type in [
        "scratch",
        "frozen",
        "scheduled",
        "full",
    ]:  # Order matters for legend
        if freeze_type not in FREEZE_TYPES:
            continue

        data = arch_data[arch_data["freeze_type"] == freeze_type]
        if len(data) == 0:
            continue

        # Sort by FLOPs
        data = data.sort_values("train_flops")

        # Collect points for frontier
        all_flops.extend(data["train_flops"].values)
        all_metrics.extend(data[metric_col].values)
        all_labels.extend([freeze_type] * len(data))

        # Plot line
        line = ax.plot(
            data["train_flops"],
            data[metric_col],
            marker="o",
            label=FREEZE_TYPES[freeze_type],
            linewidth=2.5,
            markersize=8,
            alpha=0.8,
        )[
            0
        ]  # Get the line object for color

        # Add text labels with training data percentage
        for i, (_, row) in enumerate(data.iterrows()):
            train_pct_label = f"{row['train_pct']*100:.0f}%"

            # Offset text slightly to avoid overlap with point
            ax.annotate(
                train_pct_label,
                xy=(row["train_flops"], row[metric_col]),
                xytext=(10, 5),  # Offset in points
                textcoords="offset points",
                fontsize=9,
                color=line.get_color(),  # Match line color
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=line.get_color(),
                    alpha=0.8,
                ),
            )

    # Compute and plot efficient frontier
    if len(all_flops) > 0:
        all_flops = np.array(all_flops)
        all_metrics = np.array(all_metrics)

        frontier_indices = compute_efficient_frontier(all_flops, all_metrics)

        # Sort frontier points by flops for smooth line
        frontier_flops = all_flops[frontier_indices]
        frontier_metrics = all_metrics[frontier_indices]
        sort_idx = np.argsort(frontier_flops)

        # Plot efficient frontier
        ax.plot(
            frontier_flops[sort_idx],
            frontier_metrics[sort_idx],
            "k--",
            linewidth=3,
            label="Efficient Frontier",
            alpha=0.7,
            zorder=10,
        )

        # Highlight frontier points
        ax.scatter(
            frontier_flops,
            frontier_metrics,
            s=200,
            facecolors="none",
            edgecolors="black",
            linewidth=3,
            zorder=11,
            alpha=0.8,
        )

    # Formatting
    ax.set_xlabel("Training FLOPs (log scale)", fontsize=14)
    ax.set_ylabel(METRICS[metric], fontsize=14)

    # Set log scale for x-axis
    ax.set_xscale("log")

    # Title
    arch_name = ARCH_MAPPING.get(architecture, architecture.upper())
    ax.set_title(
        f"{METRICS[metric]} vs Training FLOPs - {arch_name}\n"
        f"(100% Unlabeled Pretraining Data, {CHECKPOINT.capitalize()} Checkpoint)",
        fontsize=16,
        fontweight="bold",
    )

    # Grid
    ax.grid(True, alpha=0.3, which="both")

    # Legend
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)

    # Add minor gridlines
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()

    plt.tight_layout()

    return fig


def create_combined_architecture_plot(df, metric):
    """Create a plot with all architectures as subplots with efficient frontiers"""

    # Get unique architectures (excluding scratch)
    architectures = sorted(df[df["arch"] != "scratch"]["arch"].unique())

    if len(architectures) == 0:
        print(f"No architecture data for {metric}")
        return None

    # Create subplots
    fig, axes = plt.subplots(
        1, len(architectures), figsize=(6 * len(architectures), 6), sharey=True
    )

    if len(architectures) == 1:
        axes = [axes]

    for idx, arch in enumerate(architectures):
        ax = axes[idx]

        # Filter for specific architecture
        arch_data = df[(df["arch"] == arch) | (df["checkpoint"] == "scratch")].copy()

        metric_col = f"{metric}_{CHECKPOINT}"
        arch_data = arch_data.dropna(subset=[metric_col, "train_flops"])

        # Collect all points for efficient frontier calculation
        all_flops = []
        all_metrics = []

        # Plot each freeze type
        for freeze_type in ["scratch", "frozen", "scheduled", "full"]:
            if freeze_type not in FREEZE_TYPES:
                continue

            data = arch_data[arch_data["freeze_type"] == freeze_type]
            if len(data) == 0:
                continue

            # Sort by FLOPs
            data = data.sort_values("train_flops")

            # Collect points
            all_flops.extend(data["train_flops"].values)
            all_metrics.extend(data[metric_col].values)

            # Plot line
            line = ax.plot(
                data["train_flops"],
                data[metric_col],
                marker="o",
                label=FREEZE_TYPES[freeze_type]
                if idx == 0
                else "",  # Only label first subplot
                linewidth=2.5,
                markersize=8,
                alpha=0.8,
            )[0]

            # Add text labels
            for _, row in data.iterrows():
                train_pct_label = f"{row['train_pct']*100:.0f}%"
                ax.annotate(
                    train_pct_label,
                    xy=(row["train_flops"], row[metric_col]),
                    xytext=(10, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=line.get_color(),
                    weight="bold",
                )

        # Compute and plot efficient frontier
        if len(all_flops) > 0:
            all_flops = np.array(all_flops)
            all_metrics = np.array(all_metrics)

            frontier_indices = compute_efficient_frontier(all_flops, all_metrics)

            # Sort frontier points
            frontier_flops = all_flops[frontier_indices]
            frontier_metrics = all_metrics[frontier_indices]
            sort_idx = np.argsort(frontier_flops)

            # Plot efficient frontier
            ax.plot(
                frontier_flops[sort_idx],
                frontier_metrics[sort_idx],
                "k--",
                linewidth=2.5,
                label="Efficient Frontier" if idx == 0 else "",
                alpha=0.7,
                zorder=10,
            )

            # Highlight frontier points
            ax.scatter(
                frontier_flops,
                frontier_metrics,
                s=150,
                facecolors="none",
                edgecolors="black",
                linewidth=2.5,
                zorder=11,
                alpha=0.8,
            )

        # Formatting
        ax.set_xlabel("Training FLOPs (log scale)", fontsize=12)
        if idx == 0:
            ax.set_ylabel(METRICS[metric], fontsize=12)

        # Set log scale for x-axis
        ax.set_xscale("log")

        arch_name = ARCH_MAPPING.get(arch, arch.upper())
        ax.set_title(arch_name, fontsize=14, fontweight="bold")

        ax.grid(True, alpha=0.3, which="both")
        ax.minorticks_on()

    # Add legend to first subplot
    if len(architectures) > 0:
        axes[0].legend(loc="best", frameon=True)

    # Overall title
    fig.suptitle(
        f"{METRICS[metric]} vs Training FLOPs - All Architectures\n"
        f"(100% Unlabeled Pretraining Data, {CHECKPOINT.capitalize()} Checkpoint)",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    return fig


def main():
    # Fetch and process data
    df = fetch_runs()

    if len(df) == 0:
        print("No runs found matching criteria")
        return

    # Aggregate duplicate runs
    df_agg = aggregate_runs(df)

    # Save raw data
    df_agg.to_csv(OUTPUT_DIR / "pareto_plot_data.csv", index=False)
    print(f"\nSaved raw data to {OUTPUT_DIR / 'pareto_plot_data.csv'}")

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
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(OUTPUT_DIR / "pareto_plots.pdf") as pdf:
        for metric in METRICS:
            # Individual plots for each architecture
            for arch in architectures:
                print(f"\nCreating pareto plot for {metric} - {arch}")
                fig = create_pareto_plot(df_agg, metric, arch)
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)

            # Combined plot with all architectures
            print(f"\nCreating combined architecture plot for {metric}")
            fig = create_combined_architecture_plot(df_agg, metric)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)

    print(f"\nSaved plots to {OUTPUT_DIR / 'pareto_plots.pdf'}")


if __name__ == "__main__":
    main()
