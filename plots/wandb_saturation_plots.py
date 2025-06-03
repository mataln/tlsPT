from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb

warnings.filterwarnings("ignore")

# Initialize wandb
api = wandb.Api()

# Constants
PROJECT = "mja2106/TUNE_TLSPT_2025"
# Exact checkpoint to look for
TARGET_CHECKPOINT = "vits_radius0.2_neighbors32_mr07_uldata1.0_best_lost0.0018.ckpt"
OUTPUT_DIR = Path("saved_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Define metrics to plot
METRICS = {
    "bal_acc": "Balanced Accuracy",
    "miou": "Mean IoU",
    "acc": "Accuracy",
    "loss": "Loss",
}

CHECKPOINTS = ["first", "last", "best"]
FREEZE_TYPES = {
    "full": "Full Finetune",
    "frozen": "Frozen Encoder",
    "scratch": "From Scratch",
    "scheduled": "Scheduled Unfreeze",
}


def fetch_runs():
    """Fetch all runs from the W&B project"""
    print(f"Fetching runs from {PROJECT}...")
    print(f"Looking for checkpoint: {TARGET_CHECKPOINT} or scratch")
    runs = api.runs(PROJECT)

    data = []
    skipped_checkpoints = set()
    included_checkpoints = set()

    for run in runs:
        # Get config and summary data
        config = run.config
        summary = run.summary._json_dict

        # Skip if no ablation data
        if "ablation/freeze_type" not in config:
            continue

        # Extract checkpoint name
        checkpoint = config.get("ablation/checkpoint", "scratch")

        # Filter by exact checkpoint match
        if checkpoint != "scratch" and checkpoint != TARGET_CHECKPOINT:
            skipped_checkpoints.add(checkpoint)
            continue

        included_checkpoints.add(checkpoint)

        # Simplify checkpoint name for display
        display_checkpoint = (
            "pretrained" if checkpoint == TARGET_CHECKPOINT else "scratch"
        )

        # Extract data
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "freeze_type": config.get("ablation/freeze_type"),
            "train_pct": config.get(
                "ablation/train_pct", config.get("ablation/train_percent")
            ),
            "checkpoint": display_checkpoint,
            "run_index": config.get("ablation/run_index", 0),
        }

        # Extract metrics
        for metric in METRICS:
            for ckpt in CHECKPOINTS:
                key = f"test/{metric}_epoch_{ckpt}"
                if key in summary:
                    row[f"{metric}_{ckpt}"] = summary[key]
                else:
                    # Try alternative naming
                    alt_key = f"test/{metric}_{ckpt}"
                    if alt_key in summary:
                        row[f"{metric}_{ckpt}"] = summary[alt_key]

        data.append(row)

    df = pd.DataFrame(data)

    # Print checkpoint filtering info
    print(f"\nCheckpoint filtering summary:")
    print(f"  Included checkpoints: {included_checkpoints}")
    print(f"  Skipped {len(skipped_checkpoints)} other checkpoints")
    if len(skipped_checkpoints) <= 5:
        for ckpt in skipped_checkpoints:
            print(f"    - {ckpt}")

    print(f"\nFetched {len(df)} runs after checkpoint filtering")

    # Filter out rows with missing essential data
    df = df.dropna(subset=["freeze_type", "train_pct"])
    print(f"After filtering: {len(df)} runs")

    return df


def aggregate_runs(df):
    """Aggregate multiple runs with same parameters"""
    # Group by parameters
    group_cols = ["freeze_type", "train_pct", "checkpoint"]

    # Check if there are duplicates
    duplicates = df.groupby(group_cols).size()
    duplicates = duplicates[duplicates > 1]

    if len(duplicates) > 0:
        print("\nFound duplicate runs for the following configurations:")
        for idx, count in duplicates.items():
            print(f"  {dict(zip(group_cols, idx))}: {count} runs")

    # Aggregate
    metric_cols = [f"{m}_{c}" for m in METRICS for c in CHECKPOINTS]
    existing_cols = [col for col in metric_cols if col in df.columns]

    agg_dict = {col: "mean" for col in existing_cols}
    agg_dict["run_id"] = "count"  # Count number of runs

    df_agg = df.groupby(group_cols).agg(agg_dict).reset_index()
    df_agg.rename(columns={"run_id": "num_runs"}, inplace=True)

    return df_agg


def create_saturation_plot(df, metric, checkpoint="best"):
    """Create a saturation plot for a specific metric and checkpoint"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter data
    metric_col = f"{metric}_{checkpoint}"
    if metric_col not in df.columns:
        print(f"Warning: {metric_col} not found in data")
        return None

    plot_data = df.dropna(subset=[metric_col])

    # Plot each freeze type
    for freeze_type in FREEZE_TYPES:
        data = plot_data[plot_data["freeze_type"] == freeze_type]
        if len(data) == 0:
            continue

        # Separate by checkpoint type
        for ckpt_type in ["scratch", "pretrained"]:
            subset = data[data["checkpoint"] == ckpt_type]
            if len(subset) == 0:
                continue

            # Sort by train_pct
            subset = subset.sort_values("train_pct")

            # Determine line style
            linestyle = "-" if ckpt_type == "pretrained" else "--"
            label = f"{FREEZE_TYPES[freeze_type]} ({ckpt_type})"

            # Plot with error bars if multiple runs
            if "num_runs" in subset.columns:
                # Calculate std if we have access to individual runs
                ax.plot(
                    subset["train_pct"] * 100,
                    subset[metric_col],
                    linestyle=linestyle,
                    marker="o",
                    label=label,
                    linewidth=2,
                )
            else:
                ax.plot(
                    subset["train_pct"] * 100,
                    subset[metric_col],
                    linestyle=linestyle,
                    marker="o",
                    label=label,
                    linewidth=2,
                )

    # Formatting
    ax.set_xlabel("Training Data Percentage (%)", fontsize=12)
    ax.set_ylabel(METRICS[metric], fontsize=12)
    ax.set_title(
        f"{METRICS[metric]} vs Training Data Percentage ({checkpoint.capitalize()} Checkpoint)",
        fontsize=14,
        fontweight="bold",
    )

    # Set x-axis to percentage scale
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    # Adjust layout
    plt.tight_layout()

    return fig


def create_comparison_grid(df):
    """Create a grid of plots comparing all metrics and checkpoints"""
    metrics_to_plot = ["bal_acc", "miou"]  # Focus on main metrics

    fig, axes = plt.subplots(len(metrics_to_plot), len(CHECKPOINTS), figsize=(18, 10))

    if len(metrics_to_plot) == 1:
        axes = axes.reshape(1, -1)

    for i, metric in enumerate(metrics_to_plot):
        for j, checkpoint in enumerate(CHECKPOINTS):
            ax = axes[i, j]

            metric_col = f"{metric}_{checkpoint}"
            if metric_col not in df.columns:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            plot_data = df.dropna(subset=[metric_col])

            # Plot each configuration
            for freeze_type in FREEZE_TYPES:
                data = plot_data[plot_data["freeze_type"] == freeze_type]
                if len(data) == 0:
                    continue

                for ckpt_type in ["scratch", "pretrained"]:
                    subset = data[data["checkpoint"] == ckpt_type]
                    if len(subset) == 0:
                        continue

                    subset = subset.sort_values("train_pct")
                    linestyle = "-" if ckpt_type == "pretrained" else "--"

                    ax.plot(
                        subset["train_pct"] * 100,
                        subset[metric_col],
                        linestyle=linestyle,
                        marker="o",
                        label=f"{freeze_type} ({ckpt_type})",
                        linewidth=1.5,
                    )

            # Formatting
            ax.set_xlabel("Training Data %" if i == len(metrics_to_plot) - 1 else "")
            ax.set_ylabel(METRICS[metric] if j == 0 else "")
            ax.set_title(f"{checkpoint.capitalize()}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 105)

            # Only show legend on first plot
            if i == 0 and j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.suptitle(
        "Performance Saturation Curves Across Different Training Strategies",
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
    df_agg.to_csv(OUTPUT_DIR / "saturation_data.csv", index=False)
    print(f"\nSaved raw data to {OUTPUT_DIR / 'saturation_data.csv'}")

    # Create individual plots for main metrics
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(OUTPUT_DIR / "saturation_plots.pdf") as pdf:
        # Individual plots for best checkpoint
        for metric in ["bal_acc", "miou"]:
            fig = create_saturation_plot(df_agg, metric, "best")
            if fig:
                pdf.savefig(fig)
                plt.close(fig)

        # Comparison grid
        fig = create_comparison_grid(df_agg)
        pdf.savefig(fig)
        plt.close(fig)

        # Additional plots for other checkpoints
        for checkpoint in ["first", "last"]:
            for metric in ["bal_acc", "miou"]:
                fig = create_saturation_plot(df_agg, metric, checkpoint)
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)

    print(f"\nSaved plots to {OUTPUT_DIR / 'saturation_plots.pdf'}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for freeze_type in df_agg["freeze_type"].unique():
        print(f"\n{FREEZE_TYPES.get(freeze_type, freeze_type)}:")
        subset = df_agg[df_agg["freeze_type"] == freeze_type]
        print(f"  Data points: {len(subset)}")
        print(f"  Train percentages: {sorted(subset['train_pct'].unique() * 100)}%")

        # Best performance
        if "bal_acc_best" in subset.columns and len(subset) > 0:
            valid_subset = subset.dropna(subset=["bal_acc_best"])
            if len(valid_subset) > 0:
                best_idx = valid_subset["bal_acc_best"].idxmax()
                if pd.notna(best_idx):
                    best_bal_acc = valid_subset.loc[best_idx]
                    print(
                        f"  Best Bal Acc: {best_bal_acc['bal_acc_best']:.4f} at {best_bal_acc['train_pct']*100:.0f}% data"
                    )
                else:
                    print(f"  Best Bal Acc: No valid data")

        if "miou_best" in subset.columns and len(subset) > 0:
            valid_subset = subset.dropna(subset=["miou_best"])
            if len(valid_subset) > 0:
                best_idx = valid_subset["miou_best"].idxmax()
                if pd.notna(best_idx):
                    best_miou = valid_subset.loc[best_idx]
                    print(
                        f"  Best mIoU: {best_miou['miou_best']:.4f} at {best_miou['train_pct']*100:.0f}% data"
                    )
                else:
                    print(f"  Best mIoU: No valid data")


if __name__ == "__main__":
    main()
