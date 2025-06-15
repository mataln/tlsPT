from __future__ import annotations

import re
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
PROJECT = "mja2106/FINAL_TUNE_TLSPT_2025"
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

CHECKPOINTS = ["first", "last", "best", "step97"]
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
        return "scratch", None

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
    checkpoint_info = {}  # Track unique checkpoints
    freeze_type_counts = {}  # Debug: track freeze types

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

        # Debug: count freeze types
        freeze_type_counts[freeze_type] = freeze_type_counts.get(freeze_type, 0) + 1

        # Parse checkpoint info
        arch, uldata_pct = parse_checkpoint_name(checkpoint)

        # Track checkpoint info
        if checkpoint not in checkpoint_info:
            checkpoint_info[checkpoint] = {
                "arch": arch,
                "uldata_pct": uldata_pct,
                "count": 0,
            }
        checkpoint_info[checkpoint]["count"] += 1

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
            "run_index": config.get("ablation/run_index", 0),
        }

        # Extract metrics - try multiple possible key patterns
        for metric in METRICS:
            for ckpt in CHECKPOINTS:
                # Try different key patterns
                possible_keys = [
                    f"test/{metric}_epoch_{ckpt}",
                    f"test/{metric}_{ckpt}",
                    f"test/{metric}_{ckpt}_epoch",
                    f"test_{metric}_{ckpt}",
                    f"test_{metric}_epoch_{ckpt}",
                ]

                found = False
                for key in possible_keys:
                    if key in summary:
                        row[f"{metric}_{ckpt}"] = summary[key]
                        found = True
                        break

                # If still not found and this is a scratch run, note it
                if not found and checkpoint == "scratch" and len(data) < 3:
                    available_keys = [
                        k for k in summary.keys() if metric in k and ckpt in k
                    ]
                    if available_keys:
                        print(
                            f"    Could not find {metric}_{ckpt}, but found similar: {available_keys}"
                        )

        # Debug: For scratch runs, print what metrics were found
        if (
            checkpoint == "scratch" and len(data) < 3
        ):  # Only print for first few scratch runs
            print(f"\n  Scratch run {run.name}:")
            print(
                f"    Available test keys: {sorted([k for k in summary.keys() if 'test' in k])}"
            )
            found_metrics = [
                (k, v)
                for k, v in row.items()
                if any(m in k for m in METRICS) and v is not None
            ]
            print(f"    Extracted metrics: {found_metrics}")

        data.append(row)

    df = pd.DataFrame(data)

    # Debug output
    print(f"\nFreeze type distribution:")
    for ft, count in freeze_type_counts.items():
        print(f"  {ft}: {count} runs")

    print(f"\nScratch checkpoint analysis:")
    scratch_df = df[df["checkpoint"] == "scratch"]
    print(f"  Total scratch runs: {len(scratch_df)}")
    if len(scratch_df) > 0:
        print(
            f"  Freeze types in scratch runs: {scratch_df['freeze_type'].value_counts().to_dict()}"
        )

    # Print checkpoint summary
    print(f"\nFound {len(checkpoint_info)} unique checkpoints:")
    for ckpt, info in sorted(
        checkpoint_info.items(), key=lambda x: (x[1]["arch"], x[1]["uldata_pct"] or 0)
    ):
        if ckpt != "scratch":
            print(
                f"  {ckpt}: {info['arch']}, {info['uldata_pct']:.1f}% unlabeled data, {info['count']} runs"
            )
        else:
            print(f"  scratch: {info['count']} runs")

    print(f"\nTotal runs fetched: {len(df)}")

    # Filter out rows with missing essential data
    df = df.dropna(subset=["freeze_type", "train_pct"])
    print(f"After filtering: {len(df)} runs")

    return df, checkpoint_info


def aggregate_runs(df):
    """Aggregate multiple runs with same parameters"""
    # Debug: Check scratch data before aggregation
    print("\nScratch data before aggregation:")
    scratch_df = df[df["freeze_type"] == "scratch"]
    print(f"  Number of scratch runs: {len(scratch_df)}")
    if len(scratch_df) > 0:
        print(
            f"  Columns with data: {[col for col in scratch_df.columns if scratch_df[col].notna().any()]}"
        )
        print(f"  Sample scratch run:")
        print(scratch_df.iloc[0].to_dict())

    # Group by parameters
    group_cols = ["freeze_type", "train_pct", "checkpoint", "arch", "uldata_pct"]

    # Check if there are duplicates
    duplicates = df.groupby(group_cols, dropna=False).size()
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

    # FIXED: Use dropna=False to include NaN groups (for scratch runs with uldata_pct=NaN)
    df_agg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    df_agg.rename(columns={"run_id": "num_runs"}, inplace=True)

    # Debug: Check scratch data after aggregation
    print("\nScratch data after aggregation:")
    scratch_agg = df_agg[df_agg["freeze_type"] == "scratch"]
    print(f"  Number of scratch configurations: {len(scratch_agg)}")
    if len(scratch_agg) > 0:
        print(f"  Train percentages: {sorted(scratch_agg['train_pct'].unique())}")
        # Check which metrics have data
        for metric in METRICS:
            for ckpt in CHECKPOINTS:
                col = f"{metric}_{ckpt}"
                if col in scratch_agg.columns:
                    non_null = scratch_agg[col].notna().sum()
                    if non_null > 0:
                        print(f"  {col}: {non_null} non-null values")

    return df_agg


def create_checkpoint_comparison_grid(df, checkpoint_name, arch, uldata_pct):
    """Create a 2x3 grid comparing all metrics for a specific checkpoint"""
    metrics_to_plot = ["bal_acc", "miou"]  # Main metrics for rows

    # Debug: Check scratch data availability
    scratch_data = df[df["freeze_type"] == "scratch"]
    print(f"  Scratch data for {checkpoint_name}: {len(scratch_data)} runs")
    if len(scratch_data) > 0:
        print(f"    Train percentages: {sorted(scratch_data['train_pct'].unique())}")
        print(
            f"    Metrics available: {[col for col in scratch_data.columns if any(metric in col for metric in METRICS)]}"
        )

    fig, axes = plt.subplots(len(metrics_to_plot), len(CHECKPOINTS), figsize=(15, 8))

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

            # Debug scratch data for this specific metric/checkpoint combination
            scratch_data_all = df[df["freeze_type"] == "scratch"]
            scratch_data_metric = scratch_data_all.dropna(subset=[metric_col])

            print(f"    Plotting {metric}_{checkpoint}:")
            print(f"      Total scratch runs: {len(scratch_data_all)}")
            print(f"      Scratch runs with {metric_col}: {len(scratch_data_metric)}")
            if len(scratch_data_metric) > 0:
                print(f"      Values: {scratch_data_metric[metric_col].tolist()}")

            if checkpoint_name == "scratch":
                # For scratch page, just plot scratch data
                if len(scratch_data_metric) > 0:
                    scratch_data_metric = scratch_data_metric.sort_values("train_pct")
                    ax.plot(
                        scratch_data_metric["train_pct"] * 100,
                        scratch_data_metric[metric_col],
                        marker="o",
                        label="From Scratch",
                        linewidth=2,
                        linestyle="-",
                        color="tab:blue",
                    )
                else:
                    print(f"      WARNING: No scratch data to plot for {metric_col}!")
            else:
                # For pretrained checkpoints, plot scratch baseline first
                if len(scratch_data_metric) > 0:
                    scratch_data_metric = scratch_data_metric.sort_values("train_pct")
                    ax.plot(
                        scratch_data_metric["train_pct"] * 100,
                        scratch_data_metric[metric_col],
                        marker="o",
                        label="From Scratch",
                        linewidth=2,
                        linestyle="--",
                        color="gray",
                        alpha=0.7,
                    )

                # Then plot pretrained checkpoint data
                checkpoint_data = df[df["checkpoint"] == checkpoint_name].dropna(
                    subset=[metric_col]
                )

                for freeze_type in [
                    "full",
                    "frozen",
                    "scheduled",
                ]:  # Exclude "scratch" from this loop
                    if freeze_type not in FREEZE_TYPES:
                        continue

                    data = checkpoint_data[
                        checkpoint_data["freeze_type"] == freeze_type
                    ]
                    if len(data) == 0:
                        continue

                    # Sort by train_pct
                    data = data.sort_values("train_pct")

                    ax.plot(
                        data["train_pct"] * 100,
                        data[metric_col],
                        marker="o",
                        label=FREEZE_TYPES[freeze_type],
                        linewidth=2,
                        linestyle="-",
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

    # Create title with checkpoint info
    arch_name = ARCH_MAPPING.get(arch, arch.upper())
    if checkpoint_name == "scratch":
        title = "From Scratch (No Pretraining)"
    else:
        title = f"{arch_name} - {uldata_pct:.0f}% Unlabeled Data"

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig


def main():
    # Fetch and process data
    df, checkpoint_info = fetch_runs()

    if len(df) == 0:
        print("No runs found matching criteria")
        return

    # Aggregate duplicate runs
    df_agg = aggregate_runs(df)

    # Save raw data
    df_agg.to_csv(OUTPUT_DIR / "saturation_data_all_checkpoints.csv", index=False)
    print(f"\nSaved raw data to {OUTPUT_DIR / 'saturation_data_all_checkpoints.csv'}")

    # Sort checkpoints by architecture and unlabeled data percentage
    # Include scratch page at the beginning
    sorted_checkpoints = [
        (
            "scratch",
            checkpoint_info.get("scratch", {"arch": "scratch", "uldata_pct": None}),
        )
    ]

    # Add pretrained checkpoints
    pretrained_checkpoints = sorted(
        ((k, v) for k, v in checkpoint_info.items() if k != "scratch"),
        key=lambda x: (
            x[1]["arch"],
            x[1]["uldata_pct"] or float("inf"),  # None values go to the end
        ),
    )
    sorted_checkpoints.extend(pretrained_checkpoints)

    # Create plots for each checkpoint
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(OUTPUT_DIR / "saturation_plots_all_checkpoints.pdf") as pdf:
        for checkpoint_name, info in sorted_checkpoints:
            print(f"\nCreating plots for {checkpoint_name}")

            # Create comparison grid for this checkpoint
            fig = create_checkpoint_comparison_grid(
                df_agg, checkpoint_name, info["arch"], info["uldata_pct"]
            )
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nSaved plots to {OUTPUT_DIR / 'saturation_plots_all_checkpoints.pdf'}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")

    # First print scratch statistics
    print("\nScratch (baseline):")
    scratch_data = df_agg[df_agg["checkpoint"] == "scratch"]

    for freeze_type in scratch_data["freeze_type"].unique():
        print(f"\n  {FREEZE_TYPES.get(freeze_type, freeze_type)}:")
        subset = scratch_data[scratch_data["freeze_type"] == freeze_type]
        print(f"    Data points: {len(subset)}")
        print(f"    Train percentages: {sorted(subset['train_pct'].unique() * 100)}%")

        # Best performance
        if "bal_acc_best" in subset.columns and len(subset) > 0:
            valid_subset = subset.dropna(subset=["bal_acc_best"])
            if len(valid_subset) > 0:
                best_idx = valid_subset["bal_acc_best"].idxmax()
                if pd.notna(best_idx):
                    best_bal_acc = valid_subset.loc[best_idx]
                    print(
                        f"    Best Bal Acc: {best_bal_acc['bal_acc_best']:.4f} at {best_bal_acc['train_pct']*100:.0f}% data"
                    )

        if "miou_best" in subset.columns and len(subset) > 0:
            valid_subset = subset.dropna(subset=["miou_best"])
            if len(valid_subset) > 0:
                best_idx = valid_subset["miou_best"].idxmax()
                if pd.notna(best_idx):
                    best_miou = valid_subset.loc[best_idx]
                    print(
                        f"    Best mIoU: {best_miou['miou_best']:.4f} at {best_miou['train_pct']*100:.0f}% data"
                    )

    # Then print pretrained checkpoint statistics
    for checkpoint_name, info in pretrained_checkpoints:
        print(f"\n{checkpoint_name}:")
        checkpoint_data = df_agg[df_agg["checkpoint"] == checkpoint_name]

        if len(checkpoint_data) == 0:
            print("  No data")
            continue

        print(f"  Architecture: {ARCH_MAPPING.get(info['arch'], info['arch'])}")
        if info["uldata_pct"] is not None:
            print(f"  Unlabeled data: {info['uldata_pct']:.1f}%")

        for freeze_type in checkpoint_data["freeze_type"].unique():
            print(f"\n  {FREEZE_TYPES.get(freeze_type, freeze_type)}:")
            subset = checkpoint_data[checkpoint_data["freeze_type"] == freeze_type]
            print(f"    Data points: {len(subset)}")
            print(
                f"    Train percentages: {sorted(subset['train_pct'].unique() * 100)}%"
            )

            # Best performance
            if "bal_acc_best" in subset.columns and len(subset) > 0:
                valid_subset = subset.dropna(subset=["bal_acc_best"])
                if len(valid_subset) > 0:
                    best_idx = valid_subset["bal_acc_best"].idxmax()
                    if pd.notna(best_idx):
                        best_bal_acc = valid_subset.loc[best_idx]
                        print(
                            f"    Best Bal Acc: {best_bal_acc['bal_acc_best']:.4f} at {best_bal_acc['train_pct']*100:.0f}% data"
                        )

            if "miou_best" in subset.columns and len(subset) > 0:
                valid_subset = subset.dropna(subset=["miou_best"])
                if len(valid_subset) > 0:
                    best_idx = valid_subset["miou_best"].idxmax()
                    if pd.notna(best_idx):
                        best_miou = valid_subset.loc[best_idx]
                        print(
                            f"    Best mIoU: {best_miou['miou_best']:.4f} at {best_miou['train_pct']*100:.0f}% data"
                        )


if __name__ == "__main__":
    main()
