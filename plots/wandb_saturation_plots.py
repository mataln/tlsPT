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
        len(metrics_to_plot), len(CHECKPOINTS), figsize=(20, 8)
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
                        linewidth=2,
                        linestyle="-",
                        color="tab:blue",
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
                        linewidth=2,
                        linestyle="--",
                        color="gray",
                        alpha=0.7,
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
                        linewidth=2,
                        linestyle="-",
                        capsize=5,
                        capthick=1.5,
                        elinewidth=1.5,
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


def check_missing_runs(df_agg, df_all, filtered_runs, expected_runs=5):
    """Check for configurations with fewer than expected runs and diagnose why"""
    missing_configs = []

    # Group by the parameters we care about
    group_cols = ["freeze_type", "train_pct", "checkpoint", "arch", "uldata_pct"]

    for _, row in df_agg.iterrows():
        if row["num_runs"] < expected_runs:
            config = {col: row[col] for col in group_cols}
            config["num_runs"] = row["num_runs"]
            config["expected_runs"] = expected_runs
            config["missing_runs"] = expected_runs - row["num_runs"]
            missing_configs.append(config)

    if missing_configs:
        print(f"\n=== WARNING: Configurations with fewer than {expected_runs} runs ===")
        print(f"Total configurations with missing runs: {len(missing_configs)}")

        # Sort by number of runs and other parameters
        missing_configs.sort(
            key=lambda x: (
                x["num_runs"],
                x["checkpoint"],
                x["freeze_type"],
                x["train_pct"],
            )
        )

        for config in missing_configs:
            print(
                f"\n  Configuration: {config['checkpoint']} - {config['freeze_type']} - "
                f"{config['train_pct']*100:.0f}% data - "
                f"{config['arch']} - {config['uldata_pct']}% unlabeled"
            )
            print(
                f"    Found {config['num_runs']} runs, expected {config['expected_runs']}"
            )

            # Look for near-matches in the full dataset
            print("    Checking for near-matches in all runs...")

            # Check all runs for similar configurations
            near_matches = []
            for _, run in df_all.iterrows():
                # Check each field
                matches = {
                    "checkpoint": run.get("checkpoint") == config["checkpoint"],
                    "freeze_type": run.get("freeze_type") == config["freeze_type"],
                    "train_pct": abs(run.get("train_pct", -999) - config["train_pct"])
                    < 0.001,
                    "arch": run.get("arch") == config["arch"],
                    "uldata_pct": abs(
                        run.get("uldata_pct", -999) - config["uldata_pct"]
                    )
                    < 0.1,
                }

                # If most fields match but not all, it's a near-match
                match_count = sum(matches.values())
                if match_count >= 4 and match_count < 5:
                    mismatch_fields = [k for k, v in matches.items() if not v]
                    near_matches.append(
                        {
                            "run_id": run["run_id"],
                            "run_name": run["run_name"],
                            "mismatch_fields": mismatch_fields,
                            "values": {k: run.get(k) for k in mismatch_fields},
                        }
                    )

            if near_matches:
                print(f"    Found {len(near_matches)} near-matches:")
                for nm in near_matches[:3]:  # Show first 3
                    print(
                        f"      Run {nm['run_id']}: mismatched on {nm['mismatch_fields']}"
                    )
                    print(f"        Values: {nm['values']}")

            # Also specifically look for exact matches to understand the count
            print("    Checking exact matches in aggregated data...")
            exact_matches = df_all[
                (df_all["checkpoint"] == config["checkpoint"])
                & (df_all["freeze_type"] == config["freeze_type"])
                & (abs(df_all["train_pct"] - config["train_pct"]) < 0.001)
                & (df_all["arch"] == config["arch"])
                & (abs(df_all["uldata_pct"] - config["uldata_pct"]) < 0.1)
            ]

            if len(exact_matches) > 0:
                print(f"    Found {len(exact_matches)} exact matches:")
                for idx, match in exact_matches.iterrows():
                    print(f"      Run {match['run_id']} ({match['run_name']})")
                    print(f"        run_index: {match.get('run_index', 'unknown')}")

                # Check run indices to see which one is missing
                run_indices = sorted(exact_matches["run_index"].tolist())
                expected_indices = list(range(expected_runs))
                missing_indices = [i for i in expected_indices if i not in run_indices]
                if missing_indices:
                    print(f"    Missing run indices: {missing_indices}")
                else:
                    print(
                        f"    All run indices present but only {len(run_indices)} runs found"
                    )
            # Check filtered runs for this configuration
            print("    Checking filtered runs...")
            relevant_filtered = []
            for fr in filtered_runs:
                # Check if this filtered run might match our missing config
                if (
                    fr.get("checkpoint") == config["checkpoint"]
                    or fr.get("arch") == config["arch"]
                    or str(fr.get("uldata_pct")) == str(config["uldata_pct"])
                ):
                    relevant_filtered.append(fr)

            if relevant_filtered:
                print(
                    f"    Found {len(relevant_filtered)} potentially related filtered runs:"
                )
                for rf in relevant_filtered[:3]:  # Show first 3
                    print(f"      Run {rf['run_id']}: {rf['reason']}")
                    print(f"        Checkpoint: {rf.get('checkpoint', 'unknown')}")
            else:
                print("    No related filtered runs found")
    else:
        print(f"\n=== All configurations have at least {expected_runs} runs ===")

    return missing_configs


def main():
    # Fetch and process data
    df, checkpoint_info, filtered_runs = fetch_runs()

    if len(df) == 0:
        print("No runs found matching criteria")
        return

    # Keep a copy of the full dataframe before aggregation for diagnostics
    df_full = df.copy()

    # Aggregate duplicate runs
    df_agg = aggregate_runs(df)

    # Save raw data
    df_agg.to_csv(
        OUTPUT_DIR / "saturation_data_all_checkpoints_with_std.csv", index=False
    )
    print(
        f"\nSaved raw data to {OUTPUT_DIR / 'saturation_data_all_checkpoints_with_std.csv'}"
    )

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

    # Check for missing runs
    check_missing_runs(df_agg, df_full, filtered_runs, expected_runs=5)


if __name__ == "__main__":
    main()
