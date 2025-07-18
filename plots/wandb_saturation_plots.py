from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    "miou": "mIoU",
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

# Color palette for freeze types
FREEZE_COLORS = {
    "full": "tab:blue",
    "frozen": "tab:orange",
    "scheduled": "tab:green",
    "scratch": "gray",
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
    filtered_runs = []  # Track why runs were filtered out

    # Track all unique metric keys we find

    # Debug: sample key patterns
    sample_runs_checked = 0
    max_sample_runs = 5

    for run in runs:
        # Get config and summary data
        config = run.config
        summary = run.summary._json_dict

        # Filter out unfinished or crashed runs
        if run.state != "finished":
            filtered_runs.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "reason": f"Run state is '{run.state}' (not finished)",
                    "checkpoint": config.get("ablation/checkpoint", "unknown"),
                    "freeze_type": config.get("ablation/freeze_type", "unknown"),
                }
            )
            continue

        # Skip if no ablation data
        if "ablation/freeze_type" not in config:
            filtered_runs.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "reason": "No ablation/freeze_type in config",
                }
            )
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
                    # Additional patterns for step97
                    f"test/{metric}_at_{ckpt}",
                    f"test/{metric}_{ckpt}_checkpoint",
                    f"test/{metric}_checkpoint_{ckpt}",
                ]

                found = False
                for key in possible_keys:
                    if key in summary:
                        row[f"{metric}_{ckpt}"] = summary[key]
                        found = True
                        break

                # If still not found, note it
                if not found and sample_runs_checked < max_sample_runs:
                    available_keys = [
                        k for k in summary.keys() if metric in k and ckpt in k
                    ]
                    if available_keys:
                        print(
                            f"    Could not find {metric}_{ckpt}, but found similar: {available_keys}"
                        )

        # Debug: For first few runs, print what metrics were found
        if sample_runs_checked < max_sample_runs:
            print(f"\n  Run {run.name} ({checkpoint}, {freeze_type}):")
            print(
                f"    Available test keys: {sorted([k for k in summary.keys() if 'test' in k])}"
            )

            # Specifically check for step97 metrics
            step97_keys = [k for k in summary.keys() if "step97" in k]
            if step97_keys:
                print(f"    Step97 keys found: {step97_keys}")
                for key in step97_keys:
                    print(f"      {key}: {summary[key]}")

            found_metrics = [
                (k, v)
                for k, v in row.items()
                if any(m in k for m in METRICS) and v is not None
            ]
            print(f"    Extracted metrics: {found_metrics}")

            # Check what step97 values were extracted
            step97_extracted = [
                (k, v) for k, v in row.items() if "step97" in k and v is not None
            ]
            if step97_extracted:
                print(f"    Step97 metrics extracted: {step97_extracted}")

            sample_runs_checked += 1

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
    print(f"Runs filtered during fetch: {len(filtered_runs)}")

    # Analyze what metrics were successfully extracted
    print("\n=== Metric Extraction Summary ===")
    for metric in METRICS:
        for ckpt in CHECKPOINTS:
            col = f"{metric}_{ckpt}"
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"{col}: {non_null}/{len(df)} runs have this metric")
            else:
                print(f"{col}: Column not created (metric not found in any runs)")

    # Filter out rows with missing essential data
    df_before_filter = len(df)
    df = df.dropna(subset=["freeze_type", "train_pct"])
    if df_before_filter != len(df):
        filtered_out = df_before_filter - len(df)
        print(
            f"\nFiltered out {filtered_out} runs due to missing freeze_type or train_pct"
        )
        # Show which ones were filtered
        df_all = pd.DataFrame(data)
        missing_data = df_all[df_all["freeze_type"].isna() | df_all["train_pct"].isna()]
        for _, row in missing_data.iterrows():
            reason = []
            if pd.isna(row.get("freeze_type")):
                reason.append("missing freeze_type")
            if pd.isna(row.get("train_pct")):
                reason.append("missing train_pct")
            filtered_runs.append(
                {
                    "run_id": row["run_id"],
                    "run_name": row["run_name"],
                    "reason": ", ".join(reason),
                    "checkpoint": row.get("checkpoint", "unknown"),
                    "arch": row.get("arch", "unknown"),
                    "uldata_pct": row.get("uldata_pct", "unknown"),
                }
            )

    print(f"After filtering: {len(df)} runs")
    print(f"Total filtered runs: {len(filtered_runs)}")

    return df, checkpoint_info, filtered_runs


def aggregate_runs(df):
    """Aggregate multiple runs with same parameters, computing mean and std"""
    # Debug: Check what columns we have before aggregation
    print("\nColumns in dataframe before aggregation:")
    metric_cols = [col for col in df.columns if any(m in col for m in METRICS)]
    print(f"  Metric columns found: {sorted(metric_cols)}")

    step97_cols = [col for col in df.columns if "step97" in col]
    print(f"  Step97 columns: {step97_cols}")

    # Check how many non-null values for step97 metrics
    for col in step97_cols:
        non_null = df[col].notna().sum()
        print(f"    {col}: {non_null} non-null values out of {len(df)} rows")

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

    # Aggregate with both mean and std
    metric_cols = [f"{m}_{c}" for m in METRICS for c in CHECKPOINTS]
    existing_cols = [col for col in metric_cols if col in df.columns]

    print(f"\nColumns to aggregate: {sorted(existing_cols)}")

    # Check which expected columns are missing
    missing_cols = [col for col in metric_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns (not found in any runs): {sorted(missing_cols)}")

    # Create aggregation dictionary for mean and std
    agg_dict = {}
    for col in existing_cols:
        agg_dict[col] = ["mean", "std", "count"]
    agg_dict["run_id"] = "count"  # Count number of runs

    # FIXED: Use dropna=False to include NaN groups (for scratch runs with uldata_pct=NaN)
    df_agg = df.groupby(group_cols, dropna=False).agg(agg_dict)

    # Flatten column names
    df_agg.columns = [
        "_".join(col).strip() if col[1] else col[0] for col in df_agg.columns.values
    ]

    # Reset index
    df_agg = df_agg.reset_index()

    # Rename run count column
    df_agg.rename(columns={"run_id_count": "num_runs"}, inplace=True)

    # Debug: Check scratch data after aggregation
    print("\nScratch data after aggregation:")
    scratch_agg = df_agg[df_agg["freeze_type"] == "scratch"]
    print(f"  Number of scratch configurations: {len(scratch_agg)}")
    if len(scratch_agg) > 0:
        print(f"  Train percentages: {sorted(scratch_agg['train_pct'].unique())}")
        # Check which metrics have data
        for metric in METRICS:
            for ckpt in CHECKPOINTS:
                col = f"{metric}_{ckpt}_mean"
                if col in scratch_agg.columns:
                    non_null = scratch_agg[col].notna().sum()
                    if non_null > 0:
                        print(f"  {col}: {non_null} non-null values")

    return df_agg


def create_checkpoint_comparison_grid(df, checkpoint_name, arch, uldata_pct):
    """Create a 2x4 grid comparing all metrics for a specific checkpoint"""
    metrics_to_plot = ["bal_acc", "miou"]  # Main metrics for rows

    # Debug: Check what columns are available for this checkpoint
    print(f"\n  Creating plots for checkpoint: {checkpoint_name}")
    checkpoint_data = (
        df[df["checkpoint"] == checkpoint_name]
        if checkpoint_name != "scratch"
        else df[df["freeze_type"] == "scratch"]
    )
    available_cols = [
        col
        for col in checkpoint_data.columns
        if any(m in col for m in METRICS) and checkpoint_data[col].notna().any()
    ]
    print(f"    Available metric columns with data: {sorted(available_cols)}")

    # Specifically check step97
    step97_cols = [col for col in available_cols if "step97" in col]
    if step97_cols:
        print(f"    Step97 columns available: {step97_cols}")

    # Debug: Check scratch data availability
    scratch_data = df[df["freeze_type"] == "scratch"]
    print(f"  Scratch data for {checkpoint_name}: {len(scratch_data)} runs")
    if len(scratch_data) > 0:
        print(f"    Train percentages: {sorted(scratch_data['train_pct'].unique())}")
        print(
            f"    Metrics available: {[col for col in scratch_data.columns if any(metric in col for metric in METRICS)]}"
        )

    fig, axes = plt.subplots(
        len(metrics_to_plot), len(CHECKPOINTS), figsize=(20, 10)
    )  # Increased width for 4 columns

    if len(metrics_to_plot) == 1:
        axes = axes.reshape(1, -1)

    for i, metric in enumerate(metrics_to_plot):
        for j, checkpoint in enumerate(CHECKPOINTS):
            ax = axes[i, j]

            metric_mean_col = f"{metric}_{checkpoint}_mean"
            metric_std_col = f"{metric}_{checkpoint}_std"
            f"{metric}_{checkpoint}_count"
            print(f"      Attempting to plot {metric_mean_col}...")

            if metric_mean_col not in df.columns:
                print(f"        Column {metric_mean_col} not found in dataframe!")
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Debug scratch data for this specific metric/checkpoint combination
            scratch_data_all = df[df["freeze_type"] == "scratch"]
            scratch_data_metric = scratch_data_all.dropna(subset=[metric_mean_col])

            print(f"    Plotting {metric}_{checkpoint}:")
            print(f"      Total scratch runs: {len(scratch_data_all)}")
            print(
                f"      Scratch runs with {metric_mean_col}: {len(scratch_data_metric)}"
            )
            if len(scratch_data_metric) > 0:
                print(f"      Values: {scratch_data_metric[metric_mean_col].tolist()}")

            if checkpoint_name == "scratch":
                # For scratch page, just plot scratch data
                if len(scratch_data_metric) > 0:
                    scratch_data_metric = scratch_data_metric.sort_values("train_pct")

                    # Plot with error bars
                    x_vals = scratch_data_metric["train_pct"] * 100
                    y_vals = scratch_data_metric[metric_mean_col]
                    y_err = scratch_data_metric[metric_std_col].fillna(
                        0
                    )  # Fill NaN std with 0

                    ax.errorbar(
                        x_vals,
                        y_vals,
                        yerr=y_err,
                        marker="o",
                        label="From Scratch",
                        linewidth=2.5,
                        linestyle="-",
                        color=FREEZE_COLORS["scratch"],
                        markersize=10,
                        capsize=5,
                        capthick=1.5,
                        elinewidth=1.5,
                    )
                else:
                    print(
                        f"      WARNING: No scratch data to plot for {metric_mean_col}!"
                    )
            else:
                # For pretrained checkpoints, plot scratch baseline first
                if len(scratch_data_metric) > 0:
                    scratch_data_metric = scratch_data_metric.sort_values("train_pct")

                    # Plot with error bars
                    x_vals = scratch_data_metric["train_pct"] * 100
                    y_vals = scratch_data_metric[metric_mean_col]
                    y_err = scratch_data_metric[metric_std_col].fillna(0)

                    ax.errorbar(
                        x_vals,
                        y_vals,
                        yerr=y_err,
                        marker="o",
                        label="From Scratch",
                        linewidth=2.5,
                        linestyle="--",
                        color=FREEZE_COLORS["scratch"],
                        alpha=0.7,
                        markersize=10,
                        capsize=5,
                        capthick=1.5,
                        elinewidth=1.5,
                    )

                # Then plot pretrained checkpoint data
                checkpoint_data = df[df["checkpoint"] == checkpoint_name].dropna(
                    subset=[metric_mean_col]
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

                    # Plot with error bars
                    x_vals = data["train_pct"] * 100
                    y_vals = data[metric_mean_col]
                    y_err = data[metric_std_col].fillna(0)

                    ax.errorbar(
                        x_vals,
                        y_vals,
                        yerr=y_err,
                        marker="o",
                        label=FREEZE_TYPES[freeze_type],
                        linewidth=2.5,
                        linestyle="-",
                        color=FREEZE_COLORS[freeze_type],
                        markersize=10,
                        capsize=5,
                        capthick=1.5,
                        elinewidth=1.5,
                    )

            # Formatting
            ax.set_xlabel("Training Data %" if i == len(metrics_to_plot) - 1 else "")
            ax.set_ylabel(METRICS[metric] if j == 0 else "")
            ax.set_title(f"{checkpoint.capitalize()}", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 105)

            # Place legend inside the plot area (top left corner)
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    # Create title with checkpoint info
    arch_name = ARCH_MAPPING.get(arch, arch.upper())
    if checkpoint_name == "scratch":
        title = "From Scratch (No Pretraining)"
    else:
        title = f"{arch_name} - {uldata_pct:.0f}% Unlabeled Data"

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig


def create_publication_plot(df, metric, checkpoint_type="last"):
    """Create a single publication-quality plot for one metric"""
    fig, ax = plt.subplots(figsize=(8, 6))

    metric_mean_col = f"{metric}_{checkpoint_type}_mean"
    metric_std_col = f"{metric}_{checkpoint_type}_std"

    # Plot scratch baseline
    scratch_data = df[df["freeze_type"] == "scratch"].dropna(subset=[metric_mean_col])
    if len(scratch_data) > 0:
        scratch_data = scratch_data.sort_values("train_pct")

        x_vals = scratch_data["train_pct"] * 100
        y_vals = scratch_data[metric_mean_col]
        y_err = scratch_data[metric_std_col].fillna(0)

        ax.errorbar(
            x_vals,
            y_vals,
            yerr=y_err,
            marker="o",
            label="From Scratch",
            linewidth=2.5,
            linestyle="--",
            color=FREEZE_COLORS["scratch"],
            alpha=0.7,
            markersize=10,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
        )

    # Find checkpoint with 100% unlabeled data
    checkpoints_100 = df[(df["uldata_pct"] == 100.0) & (df["checkpoint"] != "scratch")]

    if len(checkpoints_100) > 0:
        # Get the most common checkpoint (should be the same for all)
        checkpoints_100["checkpoint"].iloc[0]
        arch = checkpoints_100["arch"].iloc[0]

        # Plot data for each freeze type
        for freeze_type in ["full", "frozen", "scheduled"]:
            if freeze_type not in FREEZE_TYPES:
                continue

            data = checkpoints_100[
                checkpoints_100["freeze_type"] == freeze_type
            ].dropna(subset=[metric_mean_col])
            if len(data) == 0:
                continue

            data = data.sort_values("train_pct")

            x_vals = data["train_pct"] * 100
            y_vals = data[metric_mean_col]
            y_err = data[metric_std_col].fillna(0)

            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                marker="o",
                label=FREEZE_TYPES[freeze_type],
                linewidth=2.5,
                linestyle="-",
                color=FREEZE_COLORS[freeze_type],
                markersize=10,
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
            )

        # Formatting
        ax.set_xlabel("Training Data %", fontsize=14)
        ax.set_ylabel(METRICS[metric], fontsize=14)
        arch_name = ARCH_MAPPING.get(arch, arch.upper())
        ax.set_title(
            f"{arch_name} - 100% Unlabeled Data ({checkpoint_type.capitalize()} Checkpoint)",
            fontsize=16,
            fontweight="bold",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No checkpoint with 100% unlabeled data found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.legend(loc="best", fontsize=12, framealpha=0.9)

    plt.tight_layout()
    return fig


def find_minimum_run_config(df_agg):
    """Find the configuration with the minimum number of runs across all data"""
    min_runs = df_agg["num_runs"].min()
    min_config = df_agg[df_agg["num_runs"] == min_runs].iloc[0]

    print("\n=== Configuration with Fewest Runs ===")
    print(f"Minimum number of runs found: {min_runs}")
    print(f"Configuration details:")
    print(f"  Checkpoint: {min_config['checkpoint']}")
    print(
        f"  Freeze type: {min_config['freeze_type']} ({FREEZE_TYPES.get(min_config['freeze_type'], min_config['freeze_type'])})"
    )
    print(f"  Train percentage: {min_config['train_pct']*100:.0f}%")
    print(f"  Architecture: {min_config['arch']}")
    if pd.notna(min_config["uldata_pct"]):
        print(f"  Unlabeled data: {min_config['uldata_pct']:.1f}%")

    # Check if there are other configurations with the same minimum
    all_min_configs = df_agg[df_agg["num_runs"] == min_runs]
    if len(all_min_configs) > 1:
        print(
            f"\nNote: {len(all_min_configs)} configurations have this minimum run count:"
        )
        for idx, config in all_min_configs.iterrows():
            print(
                f"  - {config['checkpoint']} / {config['freeze_type']} / {config['train_pct']*100:.0f}% data"
            )

    return min_config


def generate_latex_table(df_agg, checkpoint_info, output_path):
    """Generate a LaTeX table showing results across all training data percentages for 100% unlabeled data"""

    # Focus on last checkpoint and 100% unlabeled data
    eval_ckpt = "last"

    # Get unique training percentages
    train_pcts = sorted(df_agg["train_pct"].unique())

    # First, collect all data to find best values
    data_dict = {}

    for metric in ["bal_acc", "miou"]:
        data_dict[metric] = {}

        for train_pct in train_pcts:
            data_dict[metric][train_pct] = {}

            # Get scratch data
            scratch_data = df_agg[
                (df_agg["checkpoint"] == "scratch") & (df_agg["train_pct"] == train_pct)
            ]
            if len(scratch_data) > 0:
                row = scratch_data.iloc[0]
                mean_val = row[f"{metric}_{eval_ckpt}_mean"]
                std_val = row[f"{metric}_{eval_ckpt}_std"]
                if pd.notna(mean_val):
                    data_dict[metric][train_pct]["Scratch"] = (mean_val, std_val)

            # Get pretrained data
            pretrained_100 = df_agg[
                (df_agg["uldata_pct"] == 100.0)
                & (df_agg["checkpoint"] != "scratch")
                & (df_agg["train_pct"] == train_pct)
            ]

            for freeze_type, freeze_name in [
                ("full", "Full"),
                ("frozen", "Frozen"),
                ("scheduled", "Scheduled"),
            ]:
                freeze_data = pretrained_100[
                    pretrained_100["freeze_type"] == freeze_type
                ]
                if len(freeze_data) > 0:
                    row = freeze_data.iloc[0]
                    mean_val = row[f"{metric}_{eval_ckpt}_mean"]
                    std_val = row[f"{metric}_{eval_ckpt}_std"]
                    if pd.notna(mean_val):
                        data_dict[metric][train_pct][freeze_name] = (mean_val, std_val)

    # Find best values for each metric and training percentage
    best_values = {}
    for metric in ["bal_acc", "miou"]:
        best_values[metric] = {}
        for train_pct in train_pcts:
            if train_pct in data_dict[metric]:
                values = [v[0] for v in data_dict[metric][train_pct].values()]
                if values:
                    best_values[metric][train_pct] = max(values)

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("% Requires \\usepackage{multirow, booktabs, graphicx}")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append(
        "  \\caption{Performance metrics across different training data percentages using 100\\% unlabeled pretraining data (last checkpoint). Results show mean ± standard deviation. Best results per column are in bold.}"
    )
    latex_lines.append("  \\label{tab:saturation_results}")
    latex_lines.append("  \\centering")
    latex_lines.append("  \\resizebox{\\textwidth}{!}{%")

    # Create column specification
    col_spec = "ll" + "c" * len(train_pcts)
    latex_lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("    \\toprule")

    # Header row with training percentages
    header = "    Metric & Method"
    for train_pct in train_pcts:
        header += f" & {int(train_pct * 100)}\\%"
    header += " \\\\"
    latex_lines.append(header)
    latex_lines.append("    \\midrule")

    # Add data for each metric
    for metric, metric_name in [("bal_acc", "Bal. Acc"), ("miou", "mIoU")]:
        # Process each strategy
        for i, (strat_key, strat_name) in enumerate(
            [
                ("Scratch", "Scratch"),
                ("Full", "Full"),
                ("Frozen", "Frozen"),
                ("Scheduled", "Scheduled"),
            ]
        ):
            if i == 0:
                row = f"    \\multirow{{4}}{{*}}{{{metric_name}}} & {strat_name}"
            else:
                row = f"     & {strat_name}"

            for train_pct in train_pcts:
                if (
                    train_pct in data_dict[metric]
                    and strat_key in data_dict[metric][train_pct]
                ):
                    mean_val, std_val = data_dict[metric][train_pct][strat_key]
                    value_str = f"{mean_val:.3f} ± {std_val:.3f}"

                    # Bold if this is the best value
                    if (
                        train_pct in best_values[metric]
                        and mean_val == best_values[metric][train_pct]
                    ):
                        value_str = f"\\textbf{{{value_str}}}"

                    row += f" & {value_str}"
                else:
                    row += " & --"
            row += " \\\\"
            latex_lines.append(row)

        if metric != "miou":  # Add separator between metrics
            latex_lines.append("    \\midrule")

    latex_lines.append("    \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("  }%")  # Close resizebox
    latex_lines.append("\\end{table}")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"\nSaved LaTeX table to {output_path}")

    # Also create a comprehensive appendix version
    appendix_path = output_path.parent / (output_path.stem + "_appendix_full.tex")
    generate_appendix_latex_table(df_agg, checkpoint_info, appendix_path)


def generate_appendix_latex_table(df_agg, checkpoint_info, output_path):
    """Generate a comprehensive LaTeX table for appendix with all configurations"""

    # Focus on last checkpoint
    eval_ckpt = "last"

    # Get unique training percentages
    train_pcts = sorted(df_agg["train_pct"].unique())

    latex_lines = []
    latex_lines.append("% Requires \\usepackage{longtable, booktabs}")
    latex_lines.append("\\begingroup")
    latex_lines.append("\\scriptsize  % Use very small font")
    latex_lines.append("\\setlength{\\tabcolsep}{3pt}  % Minimal column spacing")
    latex_lines.append(
        "\\begin{longtable}{@{}p{1cm} p{1.5cm} p{1.5cm} p{1.5cm} p{1.5cm} p{2.5cm} p{2.5cm}@{}}"
    )
    latex_lines.append(
        "\\caption{Complete performance metrics for all training configurations. Results show mean ± standard deviation at the last checkpoint.} \\\\"
    )
    latex_lines.append("\\label{tab:saturation_results_full} \\\\")
    latex_lines.append("\\toprule")
    latex_lines.append(
        "Train \\% & Checkpoint & Arch & Unlabeled \\% & Method & Bal. Acc & mIoU \\\\"
    )
    latex_lines.append("\\midrule")
    latex_lines.append("\\endfirsthead")
    latex_lines.append("\\midrule")
    latex_lines.append("\\multicolumn{7}{r}{Continued on next page} \\\\")
    latex_lines.append("\\endfoot")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\endlastfoot")

    # Add data rows for each training percentage
    for train_pct in train_pcts:
        pct_str = f"{int(train_pct * 100)}"

        # First add scratch for this percentage
        scratch_data = df_agg[
            (df_agg["checkpoint"] == "scratch") & (df_agg["train_pct"] == train_pct)
        ]
        if len(scratch_data) > 0:
            row = scratch_data.iloc[0]
            bal_acc_mean = row[f"bal_acc_{eval_ckpt}_mean"]
            bal_acc_std = row[f"bal_acc_{eval_ckpt}_std"]
            miou_mean = row[f"miou_{eval_ckpt}_mean"]
            miou_std = row[f"miou_{eval_ckpt}_std"]

            bal_acc_str = (
                f"{bal_acc_mean:.3f} ± {bal_acc_std:.3f}"
                if pd.notna(bal_acc_mean)
                else "--"
            )
            miou_str = (
                f"{miou_mean:.3f} ± {miou_std:.3f}" if pd.notna(miou_mean) else "--"
            )

            latex_lines.append(
                f"{pct_str} & Scratch & -- & -- & Scratch & {bal_acc_str} & {miou_str} \\\\"
            )

        # Then add all pretrained configurations
        pretrained_data = df_agg[
            (df_agg["checkpoint"] != "scratch") & (df_agg["train_pct"] == train_pct)
        ]

        # Sort by architecture, unlabeled percentage, and freeze type
        if len(pretrained_data) > 0:
            pretrained_data = pretrained_data.sort_values(
                ["arch", "uldata_pct", "freeze_type"]
            )

            for _, row in pretrained_data.iterrows():
                checkpoint_name = row["checkpoint"]
                arch = ARCH_MAPPING.get(row["arch"], row["arch"].upper())
                uldata = (
                    f"{row['uldata_pct']:.0f}" if pd.notna(row["uldata_pct"]) else "--"
                )
                freeze_type = row["freeze_type"]
                freeze_name = {
                    "full": "Full",
                    "frozen": "Frozen",
                    "scheduled": "Scheduled",
                }.get(freeze_type, freeze_type)

                bal_acc_mean = row[f"bal_acc_{eval_ckpt}_mean"]
                bal_acc_std = row[f"bal_acc_{eval_ckpt}_std"]
                miou_mean = row[f"miou_{eval_ckpt}_mean"]
                miou_std = row[f"miou_{eval_ckpt}_std"]

                bal_acc_str = (
                    f"{bal_acc_mean:.3f} ± {bal_acc_std:.3f}"
                    if pd.notna(bal_acc_mean)
                    else "--"
                )
                miou_str = (
                    f"{miou_mean:.3f} ± {miou_std:.3f}" if pd.notna(miou_mean) else "--"
                )

                # Only show train % on first row of each group
                show_pct = pct_str if pretrained_data.index[0] == _ else ""

                latex_lines.append(
                    f"{show_pct} & {checkpoint_name} & {arch} & {uldata} & {freeze_name} & {bal_acc_str} & {miou_str} \\\\"
                )

        if (
            train_pct != train_pcts[-1]
        ):  # Add separator between percentages except for last
            latex_lines.append("\\midrule")

    latex_lines.append("\\end{longtable}")
    latex_lines.append("\\endgroup")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"Saved comprehensive appendix table to {output_path}")


def generate_clean_latex_table(table_data, train_pcts, output_path):
    """Generate a cleaner LaTeX table without standard deviations"""

    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append(
        "  \\caption{Performance metrics across different training data percentages using 100\\% unlabeled pretraining data (last checkpoint).}"
    )
    latex_lines.append("  \\label{tab:saturation_results_clean}")
    latex_lines.append("  \\centering")
    latex_lines.append("  \\begin{tabular}{ccccccccc}")
    latex_lines.append("    \\toprule")
    latex_lines.append(
        "    & \\multicolumn{4}{c}{Balanced Accuracy} & \\multicolumn{4}{c}{Mean IoU} \\\\"
    )
    latex_lines.append("    \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}")
    latex_lines.append(
        "    Train \\% & Scratch & Full & Frozen & Scheduled & Scratch & Full & Frozen & Scheduled \\\\"
    )
    latex_lines.append("    \\midrule")

    # Add data rows
    for train_pct in train_pcts:
        if train_pct not in table_data:
            continue

        row_data = table_data[train_pct]
        pct_str = f"{int(train_pct * 100)}"

        # Extract just the mean values
        def extract_mean(value_str):
            if value_str == "--":
                return "--"
            return value_str.split(" ±")[0]

        scratch_bal = extract_mean(row_data.get("Scratch_bal_acc", "--"))
        full_bal = extract_mean(row_data.get("Full_bal_acc", "--"))
        frozen_bal = extract_mean(row_data.get("Frozen_bal_acc", "--"))
        scheduled_bal = extract_mean(row_data.get("Scheduled_bal_acc", "--"))

        scratch_miou = extract_mean(row_data.get("Scratch_miou", "--"))
        full_miou = extract_mean(row_data.get("Full_miou", "--"))
        frozen_miou = extract_mean(row_data.get("Frozen_miou", "--"))
        scheduled_miou = extract_mean(row_data.get("Scheduled_miou", "--"))

        latex_lines.append(
            f"    {pct_str} & {scratch_bal} & {full_bal} & {frozen_bal} & {scheduled_bal} & "
            f"{scratch_miou} & {full_miou} & {frozen_miou} & {scheduled_miou} \\\\"
        )

    latex_lines.append("    \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"Saved clean LaTeX table (no std) to {output_path}")


def generate_simplified_latex_table(results_df, output_path):
    """This function is no longer needed but kept for compatibility"""


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate W&B saturation plots")
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Generate publication-ready plots (last checkpoint only, 100%% unlabeled data)",
    )
    args = parser.parse_args()

    # Set up matplotlib for publication quality
    setup_matplotlib()

    # Fetch and process data
    df, checkpoint_info, filtered_runs = fetch_runs()

    if len(df) == 0:
        print("No runs found matching criteria")
        return

    # Keep a copy of the full dataframe before aggregation for diagnostics
    df.copy()

    # Aggregate duplicate runs
    df_agg = aggregate_runs(df)

    # Save raw data
    df_agg.to_csv(
        OUTPUT_DIR / "saturation_data_all_checkpoints_with_std.csv", index=False
    )
    print(
        f"\nSaved raw data to {OUTPUT_DIR / 'saturation_data_all_checkpoints_with_std.csv'}"
    )

    # Find and report the configuration with minimum runs
    find_minimum_run_config(df_agg)

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

    if args.publication:
        # Publication mode: create separate plots for each metric
        print("\n=== Generating publication plots ===")
        print("Using last checkpoint and 100% unlabeled data only")

        for metric in ["bal_acc", "miou"]:
            fig = create_publication_plot(df_agg, metric, checkpoint_type="last")
            filename = OUTPUT_DIR / f"publication_{metric}_last_100pct.pdf"
            fig.savefig(filename, format="pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {metric} plot to {filename}")
    else:
        # Regular mode: create comprehensive plots
        # Create plots for each checkpoint
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(
            OUTPUT_DIR / "saturation_plots_all_checkpoints_with_errorbars.pdf"
        ) as pdf:
            for checkpoint_name, info in sorted_checkpoints:
                print(f"\nCreating plots for {checkpoint_name}")

                # Create comparison grid for this checkpoint
                fig = create_checkpoint_comparison_grid(
                    df_agg, checkpoint_name, info["arch"], info["uldata_pct"]
                )
                pdf.savefig(fig)
                plt.close(fig)

        print(
            f"\nSaved plots to {OUTPUT_DIR / 'saturation_plots_all_checkpoints_with_errorbars.pdf'}"
        )

    # Print summary statistics
    print("\n=== Summary Statistics ===")

    # Print which checkpoints were evaluated
    print(f"\nCheckpoints evaluated: {CHECKPOINTS}")
    print(f"Metrics evaluated: {list(METRICS.keys())}")

    # First print scratch statistics
    print("\nScratch (baseline):")
    scratch_data = df_agg[df_agg["checkpoint"] == "scratch"]

    for freeze_type in scratch_data["freeze_type"].unique():
        print(f"\n  {FREEZE_TYPES.get(freeze_type, freeze_type)}:")
        subset = scratch_data[scratch_data["freeze_type"] == freeze_type]
        print(f"    Data points: {len(subset)}")
        print(f"    Train percentages: {sorted(subset['train_pct'].unique() * 100)}%")

        # Best performance for different checkpoints
        for ckpt in ["best", "step97"]:
            if f"bal_acc_{ckpt}_mean" in subset.columns and len(subset) > 0:
                valid_subset = subset.dropna(subset=[f"bal_acc_{ckpt}_mean"])
                if len(valid_subset) > 0:
                    best_idx = valid_subset[f"bal_acc_{ckpt}_mean"].idxmax()
                    if pd.notna(best_idx):
                        best_bal_acc = valid_subset.loc[best_idx]
                        print(
                            f"    Best Bal Acc ({ckpt}): {best_bal_acc[f'bal_acc_{ckpt}_mean']:.4f} ± {best_bal_acc[f'bal_acc_{ckpt}_std']:.4f} at {best_bal_acc['train_pct']*100:.0f}% data ({best_bal_acc['num_runs']} runs)"
                        )

            if f"miou_{ckpt}_mean" in subset.columns and len(subset) > 0:
                valid_subset = subset.dropna(subset=[f"miou_{ckpt}_mean"])
                if len(valid_subset) > 0:
                    best_idx = valid_subset[f"miou_{ckpt}_mean"].idxmax()
                    if pd.notna(best_idx):
                        best_miou = valid_subset.loc[best_idx]
                        print(
                            f"    Best mIoU ({ckpt}): {best_miou[f'miou_{ckpt}_mean']:.4f} ± {best_miou[f'miou_{ckpt}_std']:.4f} at {best_miou['train_pct']*100:.0f}% data ({best_miou['num_runs']} runs)"
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

            # Best performance for different checkpoints
            for ckpt in ["best", "step97"]:
                if f"bal_acc_{ckpt}_mean" in subset.columns and len(subset) > 0:
                    valid_subset = subset.dropna(subset=[f"bal_acc_{ckpt}_mean"])
                    if len(valid_subset) > 0:
                        best_idx = valid_subset[f"bal_acc_{ckpt}_mean"].idxmax()
                        if pd.notna(best_idx):
                            best_bal_acc = valid_subset.loc[best_idx]
                            print(
                                f"    Best Bal Acc ({ckpt}): {best_bal_acc[f'bal_acc_{ckpt}_mean']:.4f} ± {best_bal_acc[f'bal_acc_{ckpt}_std']:.4f} at {best_bal_acc['train_pct']*100:.0f}% data ({best_bal_acc['num_runs']} runs)"
                            )

                if f"miou_{ckpt}_mean" in subset.columns and len(subset) > 0:
                    valid_subset = subset.dropna(subset=[f"miou_{ckpt}_mean"])
                    if len(valid_subset) > 0:
                        best_idx = valid_subset[f"miou_{ckpt}_mean"].idxmax()
                        if pd.notna(best_idx):
                            best_miou = valid_subset.loc[best_idx]
                            print(
                                f"    Best mIoU ({ckpt}): {best_miou[f'miou_{ckpt}_mean']:.4f} ± {best_miou[f'miou_{ckpt}_std']:.4f} at {best_miou['train_pct']*100:.0f}% data ({best_miou['num_runs']} runs)"
                            )

    # Generate LaTeX table with results
    latex_output_path = OUTPUT_DIR / "saturation_results_table.tex"
    generate_latex_table(df_agg, checkpoint_info, latex_output_path)


if __name__ == "__main__":
    main()
