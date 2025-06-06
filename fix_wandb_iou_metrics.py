# fix_wandb_iou_metrics.py
from __future__ import annotations

import argparse

from loguru import logger

import wandb


def clean_wandb_run_metrics(run_id, dry_run=False):
    """Clean up the IoU metrics for a single W&B run."""

    # Get run from W&B
    api = wandb.Api()
    run = api.run(f"mja2106/TUNE_TLSPT_2025/{run_id}")

    experiment_name = run.name
    logger.info(f"Processing run: {experiment_name} (ID: {run_id})")

    # Get current metrics
    current_metrics = run.summary

    # Find all IoU-related metrics
    iou_metrics = {
        k: v
        for k, v in current_metrics.items()
        if k.startswith("test/") and ("miou" in k.lower() or "iou" in k.lower())
    }

    if not iou_metrics:
        logger.info(f"No IoU metrics found in run {run_id}")
        return False

    logger.info(f"Found IoU metrics: {list(iou_metrics.keys())}")

    # Identify metrics to remove and metrics to keep
    metrics_to_remove = []
    metrics_to_keep = {}

    # Process each suffix type
    suffixes = ["_first", "_step97", "_last", "_best"]

    for suffix in suffixes:
        # Check what metrics exist for this suffix
        miou_key = f"test/miou{suffix}"
        miou_epoch_key = f"test/miou_epoch{suffix}"
        uncorrected_key = f"test/uncorrected_miou_epoch{suffix}"

        has_miou = miou_key in iou_metrics
        has_miou_epoch = miou_epoch_key in iou_metrics
        has_uncorrected = uncorrected_key in iou_metrics

        if has_miou or has_miou_epoch:
            # Determine which value to keep for miou_epoch
            if has_miou_epoch and has_uncorrected:
                # We have both - the corrected value should be in miou_epoch
                # Get the history to find the most recent value
                history = run.scan_history(keys=[miou_epoch_key])
                values = [
                    row[miou_epoch_key] for row in history if miou_epoch_key in row
                ]

                if len(values) > 1:
                    # Multiple values - take the last one (most recent)
                    corrected_value = values[-1]
                    logger.info(
                        f"{miou_epoch_key} has {len(values)} values, taking last: {corrected_value:.4f}"
                    )
                else:
                    corrected_value = iou_metrics[miou_epoch_key]
            elif has_miou and not has_miou_epoch:
                # Only has miou, use that value
                corrected_value = iou_metrics[miou_key]
            elif has_miou_epoch and not has_uncorrected:
                # Only has miou_epoch, might be an old run - skip
                logger.warning(
                    f"Run has {miou_epoch_key} but no uncorrected backup - skipping"
                )
                continue
            else:
                corrected_value = iou_metrics[miou_epoch_key]

            # Mark miou key for removal if it exists
            if has_miou:
                metrics_to_remove.append(miou_key)

            # Keep the corrected value in miou_epoch
            metrics_to_keep[miou_epoch_key] = corrected_value

            # Keep uncorrected if it exists
            if has_uncorrected:
                metrics_to_keep[uncorrected_key] = iou_metrics[uncorrected_key]

    if dry_run:
        logger.info("\n[DRY RUN] Would make the following changes:")
        logger.info(f"Metrics to remove: {metrics_to_remove}")
        logger.info(f"Metrics to keep/update: {metrics_to_keep}")
        return True

    # Apply the changes
    wandb.init(id=run_id, resume="must", project="TUNE_TLSPT_2025", entity="mja2106")

    # First, set metrics to remove to None (W&B doesn't have a delete, but None effectively removes from summary)
    removal_update = {key: None for key in metrics_to_remove}
    if removal_update:
        wandb.run.summary.update(removal_update)
        logger.info(f"Removed metrics: {list(removal_update.keys())}")

    # Then update with the correct values
    if metrics_to_keep:
        wandb.run.summary.update(metrics_to_keep)
        logger.info(f"Updated metrics: {list(metrics_to_keep.keys())}")

    # Log that we've cleaned up the metrics
    wandb.run.summary["metrics_cleaned"] = True

    wandb.finish()

    return True


def find_runs_to_clean(limit=None):
    """Find all runs that need metric cleanup."""
    api = wandb.Api()
    runs = api.runs("mja2106/TUNE_TLSPT_2025")

    runs_to_clean = []

    for i, run in enumerate(runs):
        if limit and i >= limit:
            break

        # Check if already cleaned
        if run.summary.get("metrics_cleaned", False):
            continue

        # Check if run has the correction flag (meaning it was processed by the correction script)
        if not run.summary.get("iou_correction_applied", False):
            continue

        # Check for duplicate metrics pattern
        summary_keys = list(run.summary.keys())
        has_duplicate_pattern = False

        for suffix in ["_first", "_step97", "_last", "_best"]:
            miou_key = f"test/miou{suffix}"
            miou_epoch_key = f"test/miou_epoch{suffix}"

            if miou_key in summary_keys and miou_epoch_key in summary_keys:
                has_duplicate_pattern = True
                break

        if has_duplicate_pattern:
            runs_to_clean.append(run.id)
            logger.info(f"Run {run.name} ({run.id}) needs cleanup")

    return runs_to_clean


def main():
    parser = argparse.ArgumentParser(description="Clean up W&B IoU metrics")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument("--run-id", type=str, help="Clean a specific run ID only")
    parser.add_argument("--limit", type=int, help="Limit number of runs to process")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY RUN MODE - No changes will be made ===")

    if args.run_id:
        runs_to_clean = [args.run_id]
        logger.info(f"Cleaning specific run: {args.run_id}")
    else:
        runs_to_clean = find_runs_to_clean(limit=args.limit)
        logger.info(f"Found {len(runs_to_clean)} runs to clean")

    successful = 0
    failed = 0

    for i, run_id in enumerate(runs_to_clean):
        logger.info(f"\nProcessing run {i+1}/{len(runs_to_clean)}")
        try:
            success = clean_wandb_run_metrics(run_id, dry_run=args.dry_run)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to process run {run_id}: {e}")
            failed += 1

    logger.info(f"\nSummary: {successful} successful, {failed} failed")

    if args.dry_run:
        logger.info("\n=== DRY RUN COMPLETE - No changes were made ===")


if __name__ == "__main__":
    main()
