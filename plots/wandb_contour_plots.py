from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

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

# Define metrics to plot
METRICS = {
    "bal_acc": "Balanced Accuracy",
    "miou": "Mean IoU",
}

# Fixed parameters - only full finetune and last checkpoint
FREEZE_TYPE = "full"
CHECKPOINT = "last"


def parse_checkpoint_name(checkpoint_name):
    """Extract architecture and unlabeled data percentage from checkpoint name"""
    if checkpoint_name == "scratch":
        return "scratch", 0.0  # 0% pretraining data

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

        # Only get full finetune runs
        freeze_type = config.get("ablation/freeze_type")
        if freeze_type != FREEZE_TYPE:
            continue

        # Extract checkpoint name
        checkpoint = config.get("ablation/checkpoint", "scratch")

        # Parse checkpoint info
        arch, uldata_pct = parse_checkpoint_name(checkpoint)

        # Extract data
        row = {
            "freeze_type": freeze_type,
            "train_pct": config.get(
                "ablation/train_pct", config.get("ablation/train_percent")
            ),
            "checkpoint": checkpoint,
            "arch": arch,
            "uldata_pct": uldata_pct,
            "run_index": config.get("ablation/run_index", 0),
        }

        # Extract metrics for the last checkpoint
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

    # Filter out rows with missing essential data
    df = df.dropna(subset=["freeze_type", "train_pct"])

    print(f"\nTotal full finetune runs fetched: {len(df)}")
    print(f"Runs filtered for not being finished: {filtered_count}")
    print(
        f"Unique pretraining percentages: {sorted(df['uldata_pct'].dropna().unique())}"
    )
    print(
        f"Unique downstream training percentages: {sorted(df['train_pct'].unique() * 100)}"
    )

    # Also include scratch runs for 0% pretraining
    print("\nFetching scratch runs for 0% pretraining baseline...")
    scratch_data = []

    runs = api.runs(PROJECT)  # Re-fetch to get scratch runs
    for run in runs:
        # Filter out unfinished or crashed runs
        if run.state != "finished":
            continue

        config = run.config
        summary = run.summary._json_dict

        # Only get scratch runs
        checkpoint = config.get("ablation/checkpoint", "scratch")
        freeze_type = config.get("ablation/freeze_type")

        if checkpoint != "scratch" or freeze_type != "scratch":
            continue

        row = {
            "freeze_type": "scratch",
            "train_pct": config.get(
                "ablation/train_pct", config.get("ablation/train_percent")
            ),
            "checkpoint": "scratch",
            "arch": "scratch",
            "uldata_pct": 0.0,
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

        scratch_data.append(row)

    scratch_df = pd.DataFrame(scratch_data)
    scratch_df = scratch_df.dropna(subset=["train_pct"])
    print(f"Found {len(scratch_df)} scratch runs")

    # Combine full finetune and scratch runs
    df = pd.concat([df, scratch_df], ignore_index=True)

    return df


def aggregate_runs(df):
    """Aggregate multiple runs with same parameters"""
    group_cols = ["freeze_type", "train_pct", "checkpoint", "arch", "uldata_pct"]

    metric_cols = [f"{m}_{CHECKPOINT}" for m in METRICS]
    existing_cols = [col for col in metric_cols if col in df.columns]

    agg_dict = {col: "mean" for col in existing_cols}
    agg_dict["run_index"] = "count"  # Count number of runs

    df_agg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    df_agg.rename(columns={"run_index": "num_runs"}, inplace=True)

    print(f"\nAfter aggregation: {len(df_agg)} unique configurations")

    # Check for configurations with different numbers of runs
    run_counts = df_agg["num_runs"].value_counts().sort_index()
    print("\nRun count distribution:")
    for count, freq in run_counts.items():
        print(f"  {count} runs: {freq} configurations")

    return df_agg


def create_contour_plot(df, metric, architecture):
    """Create a contour plot for a specific metric and architecture"""

    # Filter data - include both the specific architecture and scratch baseline
    data = df[(df["arch"] == architecture) | (df["checkpoint"] == "scratch")].copy()

    metric_col = f"{metric}_{CHECKPOINT}"

    # Drop rows with missing metric values
    data = data.dropna(subset=[metric_col, "uldata_pct", "train_pct"])

    if len(data) == 0:
        print(f"No data for {metric} - {architecture}")
        return None

    # Convert percentages
    pretraining_pct = data["uldata_pct"].values
    downstream_pct = data["train_pct"].values * 100
    metric_values = data[metric_col].values

    # Print data summary
    print(f"\nData points for {metric} - {architecture}:")
    unique_pretrain = sorted(set(pretraining_pct))
    unique_downstream = sorted(set(downstream_pct))
    print(f"  Pretraining %: {unique_pretrain}")
    print(f"  Downstream %: {unique_downstream}")
    print(f"  Total points: {len(data)}")

    # Separate counts for architecture vs scratch
    arch_data = data[data["arch"] == architecture]
    scratch_data = data[data["checkpoint"] == "scratch"]
    print(f"  {architecture} points: {len(arch_data)}")
    print(f"  Scratch points: {len(scratch_data)}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # For log scale, we need to handle 0% values
    pretraining_pct_plot = np.where(pretraining_pct == 0, 0.5, pretraining_pct)
    downstream_pct_plot = np.where(downstream_pct == 0, 0.5, downstream_pct)

    # Create a denser grid for smoother interpolation
    pretrain_grid = np.logspace(np.log10(0.5), np.log10(100), 50)
    downstream_grid = np.logspace(np.log10(0.5), np.log10(100), 50)

    # Create meshgrid
    X, Y = np.meshgrid(pretrain_grid, downstream_grid)

    # Interpolate the data
    Z = griddata(
        (pretraining_pct_plot, downstream_pct_plot),
        metric_values,
        (X, Y),
        method="cubic",
        fill_value=np.nan,
    )

    # Create contour plot
    contour_levels = 20
    contourf = ax.contourf(X, Y, Z, levels=contour_levels, cmap="viridis", alpha=0.8)
    contour = ax.contour(
        X, Y, Z, levels=contour_levels, colors="black", alpha=0.3, linewidths=0.5
    )

    # Add contour labels
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.3f")

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label(METRICS[metric], fontsize=12)

    # Add data points
    scatter = ax.scatter(
        pretraining_pct_plot,
        downstream_pct_plot,
        c=metric_values,
        edgecolors="black",
        linewidths=0.5,
        s=100,
        cmap="viridis",
        zorder=10,
        marker="o",
    )

    # Highlight scratch points (0% pretraining)
    scratch_mask = pretraining_pct == 0
    if np.any(scratch_mask):
        ax.scatter(
            pretraining_pct_plot[scratch_mask],
            downstream_pct_plot[scratch_mask],
            c="red",
            edgecolors="black",
            linewidths=2,
            s=150,
            zorder=11,
            marker="s",
            label="From Scratch",
        )

    # Set log scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Set axis limits
    ax.set_xlim(0.3, 150)
    ax.set_ylim(0.3, 150)

    # Custom tick labels
    xticks = [0.5, 1, 5, 10, 20, 50, 100]
    xticklabels = ["0", "1", "5", "10", "20", "50", "100"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    yticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    yticklabels = ["0", "1", "2", "5", "10", "20", "50", "100"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Labels
    ax.set_xlabel("Pretraining Data %", fontsize=14)
    ax.set_ylabel("Downstream Training Data %", fontsize=14)

    # Title
    arch_name = {"vits": "ViT-S", "vitb": "ViT-B", "vitl": "ViT-L"}.get(
        architecture, architecture.upper()
    )
    title = f"{METRICS[metric]} - Full Finetune (Last Checkpoint) - {arch_name}"
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Add grid
    ax.grid(True, alpha=0.3, which="both")

    # Add legend if we have scratch points
    if np.any(scratch_mask):
        ax.legend(loc="upper left")

    # Print value ranges for debugging
    print(
        f"  {metric} value range: [{np.min(metric_values):.4f}, {np.max(metric_values):.4f}]"
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

    # Get unique architectures (excluding scratch)
    architectures = df_agg[df_agg["arch"] != "scratch"]["arch"].unique()

    print(f"\nArchitectures found: {architectures}")

    # Create plots
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(OUTPUT_DIR / "contour_plots_full_finetune_last_averaged.pdf") as pdf:
        for metric in METRICS:
            # Create plots per architecture only
            for arch in architectures:
                print(f"\nCreating contour plot for {metric} - {arch}")
                fig = create_contour_plot(df_agg, metric, architecture=arch)
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)

    print(
        f"\nSaved contour plots to {OUTPUT_DIR / 'contour_plots_full_finetune_last_averaged.pdf'}"
    )

    # Save summary data
    summary_df = df_agg[
        ["freeze_type", "arch", "uldata_pct", "train_pct", "num_runs"]
        + [
            f"{m}_{CHECKPOINT}"
            for m in METRICS
            if f"{m}_{CHECKPOINT}" in df_agg.columns
        ]
    ]
    summary_df = summary_df.sort_values(["uldata_pct", "train_pct"])
    summary_df.to_csv(
        OUTPUT_DIR / "contour_plot_data_full_finetune_last_averaged.csv", index=False
    )
    print(
        f"\nSaved summary data to {OUTPUT_DIR / 'contour_plot_data_full_finetune_last_averaged.csv'}"
    )


if __name__ == "__main__":
    main()
