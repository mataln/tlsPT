#!/usr/bin/env python3
"""
Script to update W&B runs with corrected FLOP values based on tuning type and data percentage.
"""

from __future__ import annotations

import argparse
import re

import yaml
from loguru import logger

import wandb


def load_flops_reference(yaml_path="scheduled_flops_analysis_summary.yaml"):
    """Load the reference FLOP values for 100% data runs"""
    try:
        with open(yaml_path) as f:
            # Read the file and extract just the total_training_flops section
            content = f.read()

        # Extract the total_training_flops values using regex
        frozen_match = re.search(r"frozen_encoder:\s*([\d.e+-]+)", content)
        unfrozen_match = re.search(r"fully_unfrozen:\s*([\d.e+-]+)", content)
        scheduled_match = re.search(r"scheduled:\s*([\d.e+-]+)", content)

        if not all([frozen_match, unfrozen_match, scheduled_match]):
            # Fallback to yaml.load if regex fails
            with open(yaml_path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            flops_100pct = data["total_training_flops"]
        else:
            flops_100pct = {
                "frozen_encoder": float(frozen_match.group(1)),
                "fully_unfrozen": float(unfrozen_match.group(1)),
                "scheduled": float(scheduled_match.group(1)),
            }

    except Exception as e:
        logger.error(f"Failed to load FLOPS reference: {e}")
        raise

    # Map freeze types to their corresponding FLOP values
    freeze_type_to_flops = {
        "scratch": flops_100pct["fully_unfrozen"],  # From scratch uses full training
        "frozen": flops_100pct["frozen_encoder"],
        "scheduled": flops_100pct["scheduled"],
        "full": flops_100pct["fully_unfrozen"],
    }

    return freeze_type_to_flops


def calculate_corrected_flops(freeze_type, train_pct, flops_reference):
    """Calculate corrected FLOPs based on freeze type and training data percentage"""
    if freeze_type not in flops_reference:
        logger.warning(f"Unknown freeze type: {freeze_type}")
        return None

    # Get base FLOPs for 100% data
    base_flops = flops_reference[freeze_type]

    # Scale by training data percentage
    corrected_flops = base_flops * train_pct

    return corrected_flops


def update_wandb_run(run_id, corrected_flops, project, entity="mja2106", dry_run=False):
    """Update a single W&B run with corrected FLOPs"""
    if dry_run:
        logger.info(
            f"[DRY RUN] Would update run {run_id} with corrected_flops: {corrected_flops:.2e}"
        )
        return True

    try:
        # Resume the run to update it
        wandb.init(id=run_id, resume="must", project=project, entity=entity)

        # Log the corrected FLOPs
        wandb.log({"corrected_flops": corrected_flops})

        # Also update the summary
        wandb.run.summary["corrected_flops"] = corrected_flops

        wandb.finish()

        logger.info(
            f"Successfully updated run {run_id} with corrected_flops: {corrected_flops:.2e}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to update run {run_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update W&B runs with corrected FLOP values"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="FINAL_TUNE_TLSPT_2025",
        help="W&B project name (without entity prefix)",
    )
    parser.add_argument(
        "--entity", type=str, default="mja2106", help="W&B entity/username"
    )
    parser.add_argument(
        "--flops-yaml",
        type=str,
        default="scheduled_flops_analysis_summary.yaml",
        help="Path to YAML file with reference FLOP values",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument("--run-id", type=str, help="Update only a specific run ID")
    parser.add_argument("--limit", type=int, help="Limit number of runs to process")
    parser.add_argument(
        "--filter-updated",
        action="store_true",
        help="Skip runs that already have corrected_flops",
    )

    args = parser.parse_args()

    # Load reference FLOP values
    logger.info(f"Loading reference FLOP values from {args.flops_yaml}")
    flops_reference = load_flops_reference(args.flops_yaml)

    logger.info("Reference FLOP values (100% data):")
    for freeze_type, flops in flops_reference.items():
        logger.info(f"  {freeze_type}: {flops:.2e}")

    # Initialize W&B API
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"

    if args.run_id:
        # Process single run
        runs_to_process = [api.run(f"{project_path}/{args.run_id}")]
        logger.info(f"Processing single run: {args.run_id}")
    else:
        # Fetch all runs
        logger.info(f"Fetching runs from {project_path}...")
        runs = api.runs(project_path)
        runs_to_process = list(runs)
        logger.info(f"Found {len(runs_to_process)} total runs")

    # Statistics
    stats = {
        "total": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "no_ablation": 0,
        "already_has_flops": 0,
    }

    # Process runs
    for i, run in enumerate(runs_to_process):
        if args.limit and i >= args.limit:
            logger.info(f"Reached limit of {args.limit} runs")
            break

        stats["total"] += 1

        # Check if run has ablation data
        config = run.config
        if "ablation/freeze_type" not in config:
            stats["no_ablation"] += 1
            continue

        # Check if already has corrected_flops
        if args.filter_updated and "corrected_flops" in run.summary:
            stats["already_has_flops"] += 1
            logger.debug(f"Run {run.id} already has corrected_flops, skipping")
            continue

        # Extract freeze type and train percentage
        freeze_type = config.get("ablation/freeze_type")
        train_pct = config.get(
            "ablation/train_pct", config.get("ablation/train_percent")
        )

        if freeze_type is None or train_pct is None:
            logger.warning(f"Run {run.id} missing freeze_type or train_pct")
            stats["skipped"] += 1
            continue

        # Calculate corrected FLOPs
        corrected_flops = calculate_corrected_flops(
            freeze_type, train_pct, flops_reference
        )

        if corrected_flops is None:
            stats["skipped"] += 1
            continue

        # Log run details
        logger.info(f"\nRun: {run.name} (ID: {run.id})")
        logger.info(f"  Freeze type: {freeze_type}")
        logger.info(f"  Train percent: {train_pct * 100:.1f}%")
        logger.info(f"  Corrected FLOPs: {corrected_flops:.2e}")

        if "corrected_flops" in run.summary:
            logger.info(
                f"  Existing corrected_flops: {run.summary['corrected_flops']:.2e}"
            )

        # Update the run
        if update_wandb_run(
            run.id, corrected_flops, args.project, args.entity, args.dry_run
        ):
            stats["updated"] += 1
        else:
            stats["failed"] += 1

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total runs processed: {stats['total']}")
    logger.info(f"Runs updated: {stats['updated']}")
    logger.info(f"Runs skipped: {stats['skipped']}")
    logger.info(f"Runs failed: {stats['failed']}")
    logger.info(f"Runs without ablation data: {stats['no_ablation']}")
    if args.filter_updated:
        logger.info(
            f"Runs already having corrected_flops: {stats['already_has_flops']}"
        )

    if args.dry_run:
        logger.info("\n[DRY RUN] No changes were made to W&B")


if __name__ == "__main__":
    main()
