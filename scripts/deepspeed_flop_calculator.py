# scripts/inspect_model_flops_simple.py
from __future__ import annotations

import argparse
import io
import sys

import hydra
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_file", type=str, default="flops_profile_output.txt")
    args = parser.parse_args()

    with open(args.output_file, "w") as output_file:

        def write_output(text):
            print(text)
            output_file.write(text + "\n")
            output_file.flush()

        # Load config and create model
        write_output(f"Loading config from {args.config_path}")
        config = OmegaConf.load(args.config_path)
        model = hydra.utils.instantiate(config.model)
        model.eval()
        model.cuda()

        # Create sample batch
        batch_size = args.batch_size
        num_points = config.get("num_points", 8192)

        sample_batch = {
            "points": torch.randn(batch_size, num_points, 3).cuda(),
            "features": torch.randint(0, 2, (batch_size, num_points, 1)).float().cuda(),
            "lengths": torch.full((batch_size,), num_points).cuda(),
        }

        write_output(f"\nRunning DeepSpeed FLOPs profiler...")

        # Capture the profiler output
        captured_output = io.StringIO()
        original_stdout = sys.stdout

        try:
            # Create and run profiler
            prof = FlopsProfiler(model)
            prof.start_profile()

            with torch.no_grad():
                _ = model(sample_batch)

            prof.stop_profile()

            # Capture the detailed output
            sys.stdout = captured_output
            prof.print_model_profile(
                profile_step=1,
                module_depth=-1,  # Show all depths
                top_modules=1000,  # Show all modules
                detailed=True,
            )
            sys.stdout = original_stdout

            # Write the captured profiler output
            profile_text = captured_output.getvalue()
            output_file.write("\n" + "=" * 80 + "\n")
            output_file.write("DEEPSPEED PROFILER OUTPUT\n")
            output_file.write("=" * 80 + "\n")
            output_file.write(profile_text)
            output_file.write("\n" + "=" * 80 + "\n")

            prof.end_profile()

        except Exception as e:
            sys.stdout = original_stdout
            write_output(f"Error: {e}")

        # Also save model structure for reference
        output_file.write("\n\nMODEL PARAMETER NAMES (for schedule mapping):\n")
        output_file.write("=" * 80 + "\n")

        for name, param in model.named_parameters():
            output_file.write(f"{name}: {param.shape}\n")

    print(f"\nOutput saved to {args.output_file}")


if __name__ == "__main__":
    main()
