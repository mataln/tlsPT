# debug_model.py
from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig


def debug_model_behavior(model, dataloader, device="cuda"):
    """Run a few batches and analyze outputs in detail"""
    model.eval()
    model = model.to(device)

    all_gt_classes = []
    all_pred_classes = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get raw logits
            x_hat = model(batch)
            x_gt = batch["features"].squeeze(-1).long()
            x_pred = torch.argmax(x_hat, dim=2).long()

            print(f"\n=== Batch {i} ===")
            print(f"Logits shape: {x_hat.shape}")
            print(f"Logits range: [{x_hat.min():.3f}, {x_hat.max():.3f}]")
            print(f"Logits mean per class: {x_hat.mean(dim=(0,1))}")
            print(f"Logits std per class: {x_hat.std(dim=(0,1))}")

            # Check if logits are severely imbalanced
            probs = torch.softmax(x_hat, dim=2)
            avg_probs = probs.mean(dim=(0, 1))
            print(f"Average predicted probabilities: {avg_probs}")

            # Check actual predictions
            for j in range(x_gt.shape[0]):
                gt_unique = torch.unique(x_gt[j])
                pred_unique = torch.unique(x_pred[j])

                if len(pred_unique) == 1:
                    print(
                        f"  Item {j}: GT classes {gt_unique.cpu().numpy()}, "
                        f"Pred classes {pred_unique.cpu().numpy()} <- SINGLE CLASS!"
                    )

                # In debug script, when we see GT classes [0 1]:
                if len(gt_unique) == 2:  # Mixed class item
                    gt_class_0 = (x_gt[j] == 0).sum().item()
                    gt_class_1 = (x_gt[j] == 1).sum().item()
                    pred_class_0 = (x_pred[j] == 0).sum().item()
                    pred_class_1 = (x_pred[j] == 1).sum().item()
                    print(
                        f"    Mixed item - GT: {gt_class_0} wood, {gt_class_1} leaf | "
                        f"Pred: {pred_class_0} wood, {pred_class_1} leaf"
                    )

                all_gt_classes.append(x_gt[j].cpu())
                all_pred_classes.append(x_pred[j].cpu())

            if i >= 10:  # Check first 10 batches
                break

    # Overall statistics
    all_gt = torch.cat(all_gt_classes)
    all_pred = torch.cat(all_pred_classes)

    print(f"\n=== Overall Statistics (first 10 batches) ===")
    print(f"GT class distribution: {torch.bincount(all_gt, minlength=2)}")
    print(f"Pred class distribution: {torch.bincount(all_pred, minlength=2)}")
    print(
        f"GT class balance: {torch.bincount(all_gt, minlength=2).float() / len(all_gt)}"
    )
    print(
        f"Pred class balance: {torch.bincount(all_pred, minlength=2).float() / len(all_pred)}"
    )


@hydra.main(
    version_base="1.1",
    config_path="configs/lw_seg/",
    config_name="vits_scratch_sfp.yaml",
)
def main(config: DictConfig):
    # Initialize datamodule and prepare data FIRST
    datamodule = hydra.utils.instantiate(config.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # Your checkpoint path
    checkpoint_path = "checkpoints/tune_full_pct1.0_run1_ckptvits_radius0.2_neighbors32_mr07_uldata1.0_best_loss0.0018.ckpt_1_2025-06-06_21-29-55/last.ckpt"

    # Load checkpoint to get hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    checkpoint.get("hyper_parameters", {})

    # Get model class
    model_class = hydra.utils.get_class(config.model._target_)

    # Load with explicit parameters
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        map_location="cuda",
        strict=False,  # Ignore loss.weight mismatch
        # Required parameters from your config
        num_centers=config.model.num_centers,
        num_neighbors=config.model.num_neighbors,
        embedding_dim=config.model.embedding_dim,
        neighbor_alg=config.model.neighbor_alg,
        ball_radius=config.model.ball_radius,
        scale=config.model.scale,
        transencoder_config=config.model.transencoder_config,
        cls_dim=2,
        total_epochs=config.max_epochs,
        warmup_epochs=config.model.warmup_epochs,
        class_weights=config.model.get("class_weights", None),
    )

    val_dataloader = datamodule.val_dataloader()

    debug_model_behavior(model, val_dataloader)


if __name__ == "__main__":
    main()
