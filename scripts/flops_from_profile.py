# scripts/analyze_scheduled_flops_from_profile.py
from __future__ import annotations

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def parse_deepspeed_output(file_path):
    """Parse the DeepSpeed profiler output to extract FLOPs per module"""

    with open(file_path) as f:
        content = f.read()

    module_flops = {}

    # Extract specific module FLOPs from the aggregated profile
    patterns = [
        (r"propagation_0.*?(\d+\.?\d*)\s*GMACs", "propagation_0", 1e9),
        (r"convs1.*?(\d+\.?\d*)\s*GMACs", "convs1", 1e9),
        (r"convs2.*?(\d+\.?\d*)\s*GMACs", "convs2", 1e9),
        (r"convs3.*?(\d+\.?\d*)\s*MMACs", "convs3", 1e6),
        (r"PointNetEncoder.*?(\d+\.?\d*)\s*GMACs", "patch_encoder", 1e9),
        (r"PositionEncoder.*?(\d+\.?\d*)\s*MMACs", "pos_encoder", 1e6),
        (r"TransformerEncoder.*?(\d+\.?\d*)\s*GMACs", "transformer_encoder_total", 1e9),
    ]

    for pattern, name, scale in patterns:
        match = re.search(pattern, content)
        if match:
            macs = float(match.group(1))
            flops = macs * 2 * scale  # Convert MACs to FLOPs
            module_flops[name] = flops

    # Parse individual transformer blocks
    for i in range(12):
        module_flops[f"transformer_block_{i}"] = 376.32 * 2 * 1e6  # 0.75264 GFLOPs each

    # BatchNorm and other small ops
    module_flops["bns1"] = 0
    module_flops["bns2"] = 0
    module_flops["norm"] = 0

    return module_flops


def map_schedule_to_flops(schedule, module_flops):
    """Map schedule parameters to FLOP values"""

    stage_to_modules = {
        0: ["propagation_0", "convs1", "convs2", "convs3"],  # Decoder
        1: ["transformer_block_11"],
        2: ["transformer_block_10"],
        3: ["transformer_block_9"],
        4: ["transformer_block_8"],
        5: ["transformer_block_7"],
        6: ["transformer_block_6"],
        7: ["transformer_block_5"],
        8: ["transformer_block_4"],
        9: ["transformer_block_3"],
        10: ["transformer_block_2"],
        11: ["transformer_block_1"],
        12: ["transformer_block_0"],
        13: ["patch_encoder"],
        14: ["pos_encoder"],
    }

    stage_flops = {}
    for stage_id, modules in stage_to_modules.items():
        stage_flops[stage_id] = sum(module_flops.get(m, 0) for m in modules)

    return stage_flops


def calculate_baseline_scenarios(module_flops):
    """Calculate FLOPs for frozen encoder and fully unfrozen scenarios"""

    # Encoder components
    encoder_flops = (
        module_flops.get("patch_encoder", 0)
        + module_flops.get("pos_encoder", 0)
        + sum(module_flops.get(f"transformer_block_{i}", 0) for i in range(12))
    )

    # Decoder components
    decoder_flops = (
        module_flops.get("propagation_0", 0)
        + module_flops.get("convs1", 0)
        + module_flops.get("convs2", 0)
        + module_flops.get("convs3", 0)
    )

    total_forward = encoder_flops + decoder_flops

    # Frozen encoder: forward on all, backward ONLY on decoder
    frozen_scenario = {
        "forward": total_forward,
        "backward": decoder_flops * 2,
        "total": total_forward + decoder_flops * 2,
    }

    # Fully unfrozen: forward and backward on everything
    unfrozen_scenario = {
        "forward": total_forward,
        "backward": total_forward * 2,
        "total": total_forward * 3,
    }

    return {
        "frozen_encoder": frozen_scenario,
        "fully_unfrozen": unfrozen_scenario,
        "encoder_flops": encoder_flops,
        "decoder_flops": decoder_flops,
    }


def calculate_epoch_flops(
    schedule, stage_flops, total_forward_flops, total_epochs=300, batches_per_epoch=100
):
    """Calculate FLOPs for each epoch based on the schedule"""

    epoch_data = []

    for epoch in range(total_epochs):
        active_stages = []
        trainable_flops = 0

        for stage_id, stage_config in schedule.items():
            if stage_id == 0:  # Decoder always active
                active_stages.append(stage_id)
                trainable_flops += stage_flops.get(stage_id, 0)
            else:
                max_transition = stage_config.get("max_transition_epoch", float("inf"))
                if epoch >= max_transition:
                    active_stages.append(stage_id)
                    trainable_flops += stage_flops.get(stage_id, 0)

        # Forward pass is always on full model
        forward_flops = total_forward_flops

        # Backward pass is only on trainable parameters
        backward_flops = trainable_flops * 2

        epoch_data.append(
            {
                "epoch": epoch,
                "active_stages": active_stages,
                "trainable_flops": trainable_flops,
                "forward_flops": forward_flops,
                "backward_flops": backward_flops,
                "total_flops": forward_flops + backward_flops,
                "total_flops_per_epoch": (forward_flops + backward_flops)
                * batches_per_epoch,
            }
        )

    return pd.DataFrame(epoch_data)


def load_schedule(schedule_path):
    with open(schedule_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_output", type=str, required=True)
    parser.add_argument("--schedule_path", type=str, required=True)
    parser.add_argument("--batches_per_epoch", type=int, default=100)
    parser.add_argument("--total_epochs", type=int, default=300)
    parser.add_argument("--output", type=str, default="scheduled_flops_analysis.csv")
    args = parser.parse_args()

    # Parse DeepSpeed output
    print("Parsing DeepSpeed profiler output...")
    module_flops = parse_deepspeed_output(args.profile_output)

    print("\nExtracted module FLOPs:")
    for module, flops in sorted(module_flops.items()):
        if flops > 0:
            print(f"  {module}: {flops/1e9:.4f} GFLOPs")

    # Calculate baselines
    baselines = calculate_baseline_scenarios(module_flops)

    print(f"\nEncoder total: {baselines['encoder_flops']/1e9:.3f} GFLOPs")
    print(f"Decoder total: {baselines['decoder_flops']/1e9:.3f} GFLOPs")

    print("\n=== Baseline Scenarios (per batch) ===")
    print(f"Frozen Encoder:")
    print(f"  Forward: {baselines['frozen_encoder']['forward']/1e9:.3f} GFLOPs")
    print(
        f"  Backward: {baselines['frozen_encoder']['backward']/1e9:.3f} GFLOPs (decoder only)"
    )
    print(f"  Total: {baselines['frozen_encoder']['total']/1e9:.3f} GFLOPs")

    print(f"\nFully Unfrozen:")
    print(f"  Forward: {baselines['fully_unfrozen']['forward']/1e9:.3f} GFLOPs")
    print(
        f"  Backward: {baselines['fully_unfrozen']['backward']/1e9:.3f} GFLOPs (all params)"
    )
    print(f"  Total: {baselines['fully_unfrozen']['total']/1e9:.3f} GFLOPs")

    # Load schedule
    schedule = load_schedule(args.schedule_path)

    # Map schedule to FLOPs
    stage_flops = map_schedule_to_flops(schedule, module_flops)

    print("\nFLOPs per schedule stage:")
    for stage_id, flops in sorted(stage_flops.items()):
        print(f"  Stage {stage_id}: {flops/1e9:.4f} GFLOPs")

    # Calculate epoch-wise FLOPs
    total_forward = baselines["encoder_flops"] + baselines["decoder_flops"]
    epoch_df = calculate_epoch_flops(
        schedule, stage_flops, total_forward, args.total_epochs, args.batches_per_epoch
    )

    # Print transitions
    print("\n=== Training Schedule Transitions ===")
    transitions = []
    for stage_id, stage_config in schedule.items():
        if stage_id > 0:
            max_epoch = stage_config.get("max_transition_epoch", "N/A")
            transitions.append((max_epoch, stage_id))

    transitions.sort()
    for epoch, stage in transitions:
        print(f"Epoch {epoch}: Stage {stage} becomes trainable")

    # Calculate totals
    scheduled_total = epoch_df["total_flops_per_epoch"].sum()
    frozen_total = (
        baselines["frozen_encoder"]["total"]
        * args.batches_per_epoch
        * args.total_epochs
    )
    unfrozen_total = (
        baselines["fully_unfrozen"]["total"]
        * args.batches_per_epoch
        * args.total_epochs
    )

    print("\n=== Total Training FLOPs Comparison ===")
    print(
        f"Scheduled unfreezing: {scheduled_total:.2e} ({scheduled_total/unfrozen_total:.1%} of fully unfrozen)"
    )
    print(
        f"Frozen encoder: {frozen_total:.2e} ({frozen_total/unfrozen_total:.1%} of fully unfrozen)"
    )
    print(f"Fully unfrozen: {unfrozen_total:.2e} (100%)")
    print(f"\nScheduled vs Frozen: {(scheduled_total-frozen_total)/frozen_total:+.1%}")
    print(
        f"Scheduled vs Unfrozen: {(scheduled_total-unfrozen_total)/unfrozen_total:+.1%}"
    )

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: FLOPs per epoch
    ax1.plot(
        epoch_df["epoch"],
        epoch_df["total_flops"] / 1e9,
        "b-",
        linewidth=2,
        label="Scheduled",
    )
    ax1.axhline(
        y=baselines["frozen_encoder"]["total"] / 1e9,
        color="g",
        linestyle="--",
        label=f"Frozen Encoder ({baselines['frozen_encoder']['total']/1e9:.1f} GFLOPs)",
    )
    ax1.axhline(
        y=baselines["fully_unfrozen"]["total"] / 1e9,
        color="r",
        linestyle="--",
        label=f"Fully Unfrozen ({baselines['fully_unfrozen']['total']/1e9:.1f} GFLOPs)",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("FLOPs per Batch (GFLOPs)")
    ax1.set_title("Training FLOPs Throughout Fine-tuning Schedule")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add vertical lines for transitions
    for epoch, stage in transitions:
        ax1.axvline(x=epoch, color="gray", linestyle=":", alpha=0.5)
        ax1.text(
            epoch,
            ax1.get_ylim()[1] * 0.95,
            f"S{stage}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
        )

    # Plot 2: Cumulative FLOPs
    cumulative_scheduled = np.cumsum(epoch_df["total_flops_per_epoch"])
    epochs = np.arange(args.total_epochs)
    cumulative_frozen = (
        (epochs + 1) * baselines["frozen_encoder"]["total"] * args.batches_per_epoch
    )
    cumulative_unfrozen = (
        (epochs + 1) * baselines["fully_unfrozen"]["total"] * args.batches_per_epoch
    )

    ax2.plot(epochs, cumulative_scheduled / 1e15, "b-", linewidth=2, label="Scheduled")
    ax2.plot(
        epochs, cumulative_frozen / 1e15, "g--", linewidth=2, label="Frozen Encoder"
    )
    ax2.plot(
        epochs, cumulative_unfrozen / 1e15, "r--", linewidth=2, label="Fully Unfrozen"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cumulative FLOPs (PFLOPs)")
    ax2.set_title("Cumulative Training FLOPs")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add shaded regions for different stages
    for i, (epoch, stage) in enumerate(transitions):
        if i < len(transitions) - 1:
            ax2.axvspan(epoch, transitions[i + 1][0], alpha=0.1, color=f"C{i%10}")
        else:
            ax2.axvspan(epoch, args.total_epochs, alpha=0.1, color=f"C{i%10}")

    plt.tight_layout()
    plt.savefig(args.output.replace(".csv", "_comparison_plot.png"), dpi=150)
    print(f"\nPlot saved to {args.output.replace('.csv', '_comparison_plot.png')}")

    # Save detailed results
    epoch_df.to_csv(args.output, index=False)
    print(f"Detailed results saved to {args.output}")

    # Save summary
    summary = {
        "total_training_flops": {
            "scheduled": float(scheduled_total),
            "frozen_encoder": float(frozen_total),
            "fully_unfrozen": float(unfrozen_total),
        },
        "savings": {
            "scheduled_vs_frozen": float(
                (frozen_total - scheduled_total) / frozen_total
            ),
            "scheduled_vs_unfrozen": float(
                (unfrozen_total - scheduled_total) / unfrozen_total
            ),
        },
        "batches_per_epoch": args.batches_per_epoch,
        "total_epochs": args.total_epochs,
        "module_breakdown": {
            "encoder_gflops": float(baselines["encoder_flops"] / 1e9),
            "decoder_gflops": float(baselines["decoder_flops"] / 1e9),
        },
        "schedule_transitions": transitions,
    }

    summary_path = args.output.replace(".csv", "_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
