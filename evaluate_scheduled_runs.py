# evaluate_scheduled_runs.py
from __future__ import annotations

import argparse
import glob
import os

import hydra
import lightning.pytorch as pl
import torch
from loguru import logger
from omegaconf import OmegaConf

import wandb


def evaluate_best_checkpoint_for_run(run_id, dry_run=False):
    """Evaluate only the best checkpoint for a specific run using its W&B config."""

    # First, get the run config from W&B
    api = wandb.Api()
    run = api.run(f"mja2106/TUNE_TLSPT_2025/{run_id}")

    # The run name in W&B should be the full experiment_name (with timestamp)
    experiment_name = run.name
    logger.info(f"Run name (experiment_name): {experiment_name}")

    # Get the full config
    run_config = run.config

    # Find the checkpoint directory
    checkpoint_dir = f"checkpoints/{experiment_name}"

    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        logger.info("Checking if checkpoints are in current directory...")
        # Sometimes the checkpoints might be in the current directory
        checkpoint_dir = experiment_name
        if not os.path.exists(checkpoint_dir):
            logger.error(f"Still not found at: {checkpoint_dir}")
            return False

    # Find best checkpoint
    best_checkpoint_pattern = os.path.join(checkpoint_dir, "best_model_*.ckpt")
    best_checkpoint_files = glob.glob(best_checkpoint_pattern)

    if not best_checkpoint_files:
        logger.error(
            f"No best checkpoint found matching pattern: {best_checkpoint_pattern}"
        )
        return False

    best_checkpoint_path = max(best_checkpoint_files, key=os.path.getmtime)
    logger.info(f"Found best checkpoint: {best_checkpoint_path}")

    if dry_run:
        logger.info(f"[DRY RUN] Would evaluate checkpoint: {best_checkpoint_path}")
        logger.info(f"[DRY RUN] Run name: {run.name}")
        logger.info(f"[DRY RUN] Run ID: {run_id}")
        logger.info(
            f"[DRY RUN] Model config: {run_config.get('model', {}).get('_target_', 'Unknown')}"
        )

        # Show what metrics already exist
        existing_test_metrics = {
            k: v for k, v in run.summary.items() if k.startswith("test/")
        }
        if existing_test_metrics:
            logger.info(
                f"[DRY RUN] Existing test metrics: {list(existing_test_metrics.keys())}"
            )

        return True

    # Resume the existing run
    wandb.init(id=run_id, resume="must", project="TUNE_TLSPT_2025", entity="mja2106")

    logger.info(f"Resumed run {run_id}")

    # Recreate datamodule from the run's config
    datamodule_config = OmegaConf.create(run_config["datamodule"])
    datamodule = hydra.utils.instantiate(datamodule_config)

    # Create trainer for testing
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Load model from checkpoint
    # Extract model params from run config
    model_params = {
        "ball_radius": run_config["model"]["ball_radius"],
        "scale": run_config["model"]["scale"],
        "neighbor_alg": run_config["model"]["neighbor_alg"],
        "num_centers": run_config["model"]["num_centers"],
        "num_neighbors": run_config["model"]["num_neighbors"],
    }

    from tlspt.models.pointmae.pointmae_seg import PointMAESegmentation

    model = PointMAESegmentation.load_from_checkpoint(
        best_checkpoint_path, **model_params
    )

    # Test
    test_results = trainer.test(model, datamodule, verbose=False)

    # Log results
    if test_results and len(test_results) > 0:
        test_metrics = test_results[0]

        # Format metrics with _best suffix
        logged_metrics = {}
        for key, value in test_metrics.items():
            clean_key = key.replace("test/", "")
            new_key = f"test/{clean_key}_best"
            logged_metrics[new_key] = value

        # Log to W&B
        wandb.log(logged_metrics)
        logger.info(f"Logged best checkpoint test metrics: {logged_metrics}")

    wandb.finish()
    return True


def find_scheduled_runs_missing_best_eval(dry_run=False):
    """Find scheduled runs that have first/last but not best evaluations."""
    api = wandb.Api()
    runs = api.runs("mja2106/TUNE_TLSPT_2025")

    runs_to_evaluate = []
    for run in runs:
        # Check if this is a scheduled run
        if run.config.get("tune_schedule") is None:
            continue

        # Check if it has first/last but not best
        summary_keys = list(run.summary.keys())
        has_first = any(
            "_first" in key for key in summary_keys if key.startswith("test/")
        )
        has_last = any(
            "_last" in key for key in summary_keys if key.startswith("test/")
        )
        has_best = any(
            "_best" in key for key in summary_keys if key.startswith("test/")
        )

        if has_first and has_last and not has_best:
            if dry_run:
                logger.info(f"\n[DRY RUN] Run needs evaluation:")
                logger.info(f"  - Name: {run.name}")
                logger.info(f"  - ID: {run.id}")
                logger.info(
                    f"  - Has test metrics: {[k for k in summary_keys if k.startswith('test/')][:5]}..."
                )  # Show first 5
            else:
                logger.info(
                    f"Run {run.id} ({run.name}) needs best checkpoint evaluation"
                )
            runs_to_evaluate.append(run.id)

    return runs_to_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate best checkpoints for scheduled runs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument("--run-id", type=str, help="Evaluate a specific run ID only")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY RUN MODE - No changes will be made ===")

    if args.run_id:
        # Evaluate specific run
        runs_to_evaluate = [args.run_id]
        logger.info(f"Evaluating specific run: {args.run_id}")
    else:
        # Find all runs needing evaluation
        runs_to_evaluate = find_scheduled_runs_missing_best_eval(dry_run=args.dry_run)
        logger.info(
            f"Found {len(runs_to_evaluate)} runs needing best checkpoint evaluation"
        )

    successful = 0
    failed = 0

    for run_id in runs_to_evaluate:
        try:
            if evaluate_best_checkpoint_for_run(run_id, dry_run=args.dry_run):
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


if __name__ == "__main__":
    main()
