# reevaluate_with_corrected_iou.py
from __future__ import annotations

import argparse
import glob
import os

import hydra
import torch
from loguru import logger
from omegaconf import OmegaConf

import wandb


def evaluate_checkpoint_iou_only(checkpoint_path, model_params, datamodule):
    """Evaluate only IoU metrics for a checkpoint."""
    from tlspt.models.pointmae.pointmae_seg import PointMAESegmentation

    try:
        model = PointMAESegmentation.load_from_checkpoint(
            checkpoint_path, **model_params
        )
        model.eval()
        model = model.cuda() if torch.cuda.is_available() else model

        # Prepare the datamodule
        datamodule.prepare_data()
        datamodule.setup(stage="test")

        # Manually compute IoU over test set
        total_miou = 0
        num_batches = 0

        with torch.no_grad():
            for batch in datamodule.test_dataloader():
                # Move batch to device
                batch = {
                    k: v.cuda() if torch.cuda.is_available() else v
                    for k, v in batch.items()
                }

                # Get predictions
                x_hat = model(batch)  # Logits (batch, N, cls_dim)
                x_pred = torch.argmax(
                    x_hat, dim=2
                ).long()  # Predicted classes (batch, N)
                x_gt = batch["features"].squeeze(-1).long()  # Ground truth (batch, N)

                # Use the model's corrected IoU calculation
                batch_miou = model.get_miou(x_pred, x_gt)
                total_miou += batch_miou.item()
                num_batches += 1

        final_miou = total_miou / num_batches if num_batches > 0 else 0
        return final_miou

    except Exception as e:
        logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
        return None


def evaluate_all_checkpoints_for_run(run_id, dry_run=False, damp_run=False):
    """Re-evaluate only IoU metrics for all checkpoints in a run."""

    # Get run config from W&B
    api = wandb.Api()
    run = api.run(f"mja2106/TUNE_TLSPT_2025/{run_id}")

    experiment_name = run.name
    logger.info(f"Processing run: {experiment_name} (ID: {run_id})")

    # Get the full config
    run_config = run.config

    # Find checkpoint directory
    checkpoint_dir = f"checkpoints/{experiment_name}"
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = experiment_name
        if not os.path.exists(checkpoint_dir):
            logger.error(f"Checkpoint directory not found for run {run_id}")
            return False

    # Define checkpoint configs
    checkpoint_configs = [
        {
            "name": "first_epoch",
            "path": os.path.join(checkpoint_dir, "first.ckpt"),
            "suffix": "_first",
        },
        {
            "name": "step_97",
            "path": os.path.join(checkpoint_dir, "step_97.ckpt"),
            "suffix": "_step97",
        },
        {
            "name": "last_epoch",
            "path": os.path.join(checkpoint_dir, "last.ckpt"),
            "suffix": "_last",
        },
    ]

    # Find best checkpoint
    best_checkpoint_pattern = os.path.join(checkpoint_dir, "best_model_*.ckpt")
    best_checkpoint_files = glob.glob(best_checkpoint_pattern)
    if best_checkpoint_files:
        best_checkpoint_path = max(best_checkpoint_files, key=os.path.getmtime)
        checkpoint_configs.append(
            {
                "name": "best_model",
                "path": best_checkpoint_path,
                "suffix": "_best",
            }
        )

    # Check which checkpoints exist
    existing_checkpoints = [
        cfg for cfg in checkpoint_configs if os.path.exists(cfg["path"])
    ]

    if not existing_checkpoints:
        logger.warning(f"No checkpoints found for run {run_id}")
        return False

    # Get current IoU metrics
    current_iou_metrics = {
        k: v
        for k, v in run.summary.items()
        if k.startswith("test/") and ("miou" in k.lower() or "iou" in k.lower())
    }

    if dry_run:
        logger.info(
            f"[DRY RUN] Would re-evaluate IoU for {len(existing_checkpoints)} checkpoints:"
        )
        for cfg in existing_checkpoints:
            logger.info(f"  - {cfg['name']}: {cfg['path']}")

        if current_iou_metrics:
            logger.info(f"\n[DRY RUN] Current IoU metrics that would be updated:")
            for key, value in sorted(current_iou_metrics.items()):
                logger.info(f"  - {key}: {value:.4f}")

        return True

    # Recreate datamodule
    datamodule_config = OmegaConf.create(run_config["datamodule"])
    datamodule = hydra.utils.instantiate(datamodule_config)

    # Extract model params
    model_params = {
        "ball_radius": run_config["model"]["ball_radius"],
        "scale": run_config["model"]["scale"],
        "neighbor_alg": run_config["model"]["neighbor_alg"],
        "num_centers": run_config["model"]["num_centers"],
        "num_neighbors": run_config["model"]["num_neighbors"],
    }

    # Evaluate each checkpoint
    iou_updates = {}
    comparison_data = []

    for cfg in existing_checkpoints:
        logger.info(f"Computing corrected IoU for {cfg['name']} checkpoint...")
        corrected_miou = evaluate_checkpoint_iou_only(
            cfg["path"], model_params, datamodule
        )

        if corrected_miou is not None:
            # Original metric names
            miou_key = f"test/miou{cfg['suffix']}"
            miou_epoch_key = f"test/miou_epoch{cfg['suffix']}"

            # Get old values if they exist
            old_miou = current_iou_metrics.get(miou_key, None)
            current_iou_metrics.get(miou_epoch_key, None)

            comparison_data.append(
                {
                    "checkpoint": cfg["name"],
                    "old_miou": old_miou,
                    "new_miou": corrected_miou,
                    "difference": corrected_miou - old_miou if old_miou else None,
                }
            )

            iou_updates[miou_key] = corrected_miou
            iou_updates[miou_epoch_key] = corrected_miou

            logger.info(f"  - Completed: {corrected_miou:.4f}")
            if old_miou:
                logger.info(
                    f"    (was: {old_miou:.4f}, difference: {corrected_miou - old_miou:+.4f})"
                )
        else:
            logger.warning(f"  - Failed to evaluate {cfg['name']}")

    if damp_run:
        logger.info("\n[DAMP RUN] Computed new IoU values but NOT updating W&B:")
        logger.info("-" * 60)
        logger.info(
            f"{'Checkpoint':<15} {'Old IoU':<10} {'New IoU':<10} {'Difference':<10}"
        )
        logger.info("-" * 60)
        for comp in comparison_data:
            old_str = f"{comp['old_miou']:.4f}" if comp["old_miou"] else "N/A"
            diff_str = (
                f"{comp['difference']:+.4f}"
                if comp["difference"] is not None
                else "N/A"
            )
            logger.info(
                f"{comp['checkpoint']:<15} {old_str:<10} {comp['new_miou']:.4f}{'':>6} {diff_str:<10}"
            )
        return True

    # Resume W&B run and update metrics
    wandb.init(id=run_id, resume="must", project="TUNE_TLSPT_2025", entity="mja2106")
    logger.info(f"Resumed run {run_id}")

    # First, preserve old IoU metrics
    preserved_metrics = {}
    for key, value in current_iou_metrics.items():
        new_key = key.replace("test/", "test/uncorrected_")
        preserved_metrics[new_key] = value

    if preserved_metrics:
        wandb.log(preserved_metrics)
        logger.info(
            f"Preserved {len(preserved_metrics)} original IoU metrics with 'uncorrected_' prefix"
        )

    # Update with corrected IoU values
    if iou_updates:
        wandb.log(iou_updates)
        logger.info(f"Updated {len(iou_updates)} IoU metrics with corrected values")

        # Log a flag indicating IoU correction
        wandb.log({"iou_correction_applied": True})

    wandb.finish()
    return True


def find_runs_to_reevaluate(dry_run=False, limit=None):
    """Find all runs that have IoU metrics and haven't been corrected yet."""
    api = wandb.Api()
    runs = api.runs("mja2106/TUNE_TLSPT_2025")

    runs_to_evaluate = []
    for i, run in enumerate(runs):
        if limit and i >= limit:
            break

        # Check if run has any IoU metrics
        summary_keys = list(run.summary.keys())
        has_iou_metrics = any(
            k.startswith("test/") and ("miou" in k.lower() or "iou" in k.lower())
            for k in summary_keys
        )

        # Check if we've already corrected
        has_correction_flag = run.summary.get("iou_correction_applied", False)
        has_uncorrected_backup = any(
            "uncorrected_" in key and ("miou" in key.lower() or "iou" in key.lower())
            for key in summary_keys
            if key.startswith("test/")
        )

        if has_iou_metrics and not has_correction_flag and not has_uncorrected_backup:
            if dry_run:
                iou_metrics = [
                    k
                    for k in summary_keys
                    if k.startswith("test/")
                    and ("miou" in k.lower() or "iou" in k.lower())
                ]
                logger.info(f"\n[DRY RUN] Run needs IoU re-evaluation:")
                logger.info(f"  - Name: {run.name}")
                logger.info(f"  - ID: {run.id}")
                logger.info(f"  - IoU metrics to update: {iou_metrics}")
            runs_to_evaluate.append(run.id)

    return runs_to_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate runs with corrected IoU calculation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--damp-run",
        action="store_true",
        help="Compute new IoU values but don't update W&B",
    )
    parser.add_argument("--run-id", type=str, help="Re-evaluate a specific run ID only")
    parser.add_argument("--limit", type=int, help="Limit number of runs to process")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY RUN MODE - No changes will be made ===")
    elif args.damp_run:
        logger.info("=== DAMP RUN MODE - Will compute values but not update W&B ===")

    if args.run_id:
        runs_to_evaluate = [args.run_id]
        logger.info(f"Re-evaluating specific run: {args.run_id}")
    else:
        runs_to_evaluate = find_runs_to_reevaluate(
            dry_run=args.dry_run, limit=args.limit
        )
        logger.info(f"Found {len(runs_to_evaluate)} runs to re-evaluate")

    successful = 0
    failed = 0

    for i, run_id in enumerate(runs_to_evaluate):
        logger.info(f"\nProcessing run {i+1}/{len(runs_to_evaluate)}")
        try:
            if evaluate_all_checkpoints_for_run(
                run_id, dry_run=args.dry_run, damp_run=args.damp_run
            ):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to evaluate run {run_id}: {e}")
            failed += 1
            continue

    logger.info(f"\nSummary: {successful} successful, {failed} failed")

    if args.dry_run:
        logger.info("\n=== DRY RUN COMPLETE - No changes were made ===")
    elif args.damp_run:
        logger.info(
            "\n=== DAMP RUN COMPLETE - Computed values but didn't update W&B ==="
        )


if __name__ == "__main__":
    main()
