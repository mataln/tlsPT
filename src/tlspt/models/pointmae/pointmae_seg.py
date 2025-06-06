from __future__ import annotations

import math

import lightning as L
import torch
from loguru import logger
from torch import nn
from torchmetrics import Accuracy
from torchmetrics.segmentation import MeanIoU

from tlspt.models.transformer import TransformerEncoder

from .components import (
    Group,
    PointNetEncoder,
    PointNetFeaturePropagation,
    PositionEncoder,
)


class PointMAESegmentation(L.LightningModule):
    def __init__(
        self,
        backbone=None,
        num_centers: int = 64,
        num_neighbors: int = 32,
        embedding_dim: int = 384,
        neighbor_alg: str = "ball_query",
        ball_radius: float = None,
        scale: float = None,
        transencoder_config: dict = {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "qkv_bias": False,
            "qk_scale": None,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
        feature_blocks: list = [3, 7, 11],
        cls_dim: int = 2,  # Leaf/Wood
        total_epochs: int = 300,
        warmup_epochs: int = 10,
        prop_mlp_dim: int = 1024,
        freeze_encoder: bool = False,
        learning_rate: float = 0.001,
        class_weights: list = None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.transencoder_config = transencoder_config
        self.trans_dim = transencoder_config["embed_dim"]
        self.pos_encoder = PositionEncoder(transformer_dim=self.trans_dim)
        logger.debug(f"Position encoder: {self.pos_encoder.state_dict().keys()}")
        self.patch_encoder = PointNetEncoder(embedding_dim=self.embedding_dim)
        logger.debug(f"Patch encoder: {self.patch_encoder.state_dict().keys()}")
        self.transformer_encoder = TransformerEncoder(**self.transencoder_config)
        logger.debug(
            f"Transformer encoder: {self.transformer_encoder.state_dict().keys()}"
        )

        if backbone is not None:  # Preload
            logger.info(
                "Loading model from pretrained backbone, other model parameters will be ignored"
            )

            self.embedding_dim = backbone[
                "transformer_encoder.blocks.0.attn.proj.weight"
            ].shape[
                0
            ]  # Needed for feature prop
            logger.info(
                f"Embedding/transformer dim from backbone: {self.embedding_dim}"
            )

            # Posn encoder
            pos_encoder_state = {
                k.replace("pos_encoder.", ""): backbone[k]
                for k in backbone.keys()
                if k.startswith("pos_encoder")
            }
            if pos_encoder_state.keys() != self.pos_encoder.state_dict().keys():
                logger.error(
                    f"Mismatch in state dicts for pos encoder: {self.pos_encoder.state_dict().keys()}"
                )
                raise ValueError("Position encoder state dict keys do not match")
            self.pos_encoder.load_state_dict(pos_encoder_state)
            logger.info(f"Loaded pos encoder from backbone")

            # Patch encoder
            patch_encoder_state = {
                k.replace("patch_encoder.", ""): backbone[k]
                for k in backbone.keys()
                if k.startswith("patch_encoder")
            }
            if patch_encoder_state.keys() != self.patch_encoder.state_dict().keys():
                logger.error(
                    f"Mismatch in state dicts for patch encoder: {self.patch_encoder.state_dict().keys()}"
                )
                raise ValueError("Patch encoder state dict keys do not match")
            self.patch_encoder.load_state_dict(patch_encoder_state)
            logger.info(f"Loaded patch encoder from backbone")

            # Transformer encoder
            transformer_encoder_state = {
                k.replace("transformer_encoder.", ""): backbone[k]
                for k in backbone.keys()
                if k.startswith("transformer_encoder")
            }
            if (
                transformer_encoder_state.keys()
                != self.transformer_encoder.state_dict().keys()
            ):
                logger.error(
                    f"Mismatch in state dicts for transformer encoder: {self.transformer_encoder.state_dict().keys()}"
                )
                raise ValueError("Transformer encoder state dict keys do not match")
            self.transformer_encoder.load_state_dict(transformer_encoder_state)
            logger.info(f"Loaded transformer encoder from backbone")

        self.freeze_encoder = freeze_encoder
        self.learning_rate = learning_rate

        self.neighbor_alg = neighbor_alg
        self.ball_radius = ball_radius
        self.num_centers = num_centers  # Can be different to backbone num_centers
        self.num_neighbors = num_neighbors  # Can be different to backbone num_neighbors
        self.scale = scale  # Can be different to backbone scale
        self.group = Group(
            num_centers=self.num_centers,
            num_neighbors=self.num_neighbors,
            neighbor_alg=self.neighbor_alg,
            radius=self.ball_radius / self.scale
            if self.neighbor_alg == "ball_query"
            else None,
        )

        if class_weights is not None:
            logger.info(f"Using class weights: {class_weights} for CrossEntropyLoss")
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            logger.info("Using CrossEntropyLoss without class weights")
            self.loss = nn.CrossEntropyLoss()

        self.miou = MeanIoU(num_classes=cls_dim, input_format="index")
        self.accuracy = Accuracy(
            task="binary" if cls_dim == 2 else "multiclass",
            num_classes=cls_dim if cls_dim > 2 else None,
        )
        self.balanced_accuracy = Accuracy(
            task="multiclass", num_classes=cls_dim, average="macro"
        )

        self.warmup_epochs = min(warmup_epochs, total_epochs)
        self.warmup_epochs = max(self.warmup_epochs, 1)

        self.cls_dim = cls_dim
        self.total_epochs = total_epochs
        self.feature_blocks = feature_blocks
        self.prop_mlp_dim = prop_mlp_dim

        # Decoder
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=len(feature_blocks) * self.embedding_dim + 3,
            mlp=[self.embedding_dim * 4, self.prop_mlp_dim],
        )

        in_dim = len(self.feature_blocks) * self.embedding_dim * 2 + self.prop_mlp_dim
        self.convs1 = nn.Conv1d(in_dim, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def reconstruct(
        self, x
    ):  # Reconstructs points from decoded patch embeddings/tokens. x is (batch, no mask centers, transformer_dim)
        B, M, _ = x.shape
        x = x.transpose(1, 2)  # x is now (batch, transformer_dim, no mask centers)
        x = self.reconstructor(x)  # x is now (batch, 3*no neighbors, no mask centers)
        x = x.transpose(1, 2)  # x is now (batch, no mask centers, 3*no neighbors)
        x = x.reshape(B, M, -1, 3)  # x is now (batch, no mask centers, no neighbors, 3)
        return x

    def forward_encoder(self, patches, centers):
        patch_embeddings = self.patch_encoder(
            patches
        )  # Embeddings for visible patches. (batch, [1-m]*centers, embedding_dim)
        pos_embeddings = self.pos_encoder(
            centers
        )  # Position embeddings for visible patches. (batch, [1-m]*centers, transformer_dim)

        # Feed visible patches and position embeddings to transformer
        x, feature_list = self.transformer_encoder(
            patch_embeddings, pos_embeddings, feature_blocks=self.feature_blocks
        )

        return pos_embeddings, feature_list

    def forward_decoder(self, x, full_pos_embeddings, N):
        x_rec = self.transformer_decoder(
            x, full_pos_embeddings, N
        )  # Reconstructed mask tokens (patch embeddings). (batch, no mask centers, transformer_dim)
        x_hat = self.reconstruct(
            x_rec
        )  # Reconstructed points. (batch, no mask centers, no neighbors, 3)

        return x_hat

    def forward(self, x):
        B, N, _ = x[
            "points"
        ].shape  # B: batch size, N: number of points, _: number of dimensions (3)

        if x["features"].shape[2] > 1:
            raise ValueError(
                "Multiple features not supported. Need to use label as only feature."
            )

        patches, centers = self.group(
            x["points"], x["lengths"]
        )  # patches (batch, no centers, no neighbors, 3), centers (batch, no centers, 3)

        # Encode
        pos_embeddings, feature_list = self.forward_encoder(
            patches, centers
        )  # x: (B, centers, transformer_dim), pos_embeddings: (batch, centers, transformer_dim), feature_tensor: no groups long list of (B, no. centers, transformer dim)

        # Stack feature list
        feature_tensor = torch.cat(
            feature_list, dim=2
        )  # Concatenate along feature dimension
        feature_tensor = feature_tensor.transpose(
            1, 2
        )  # Make it [B, C, no centers] #C: transformer dim, no. centers

        x_max = torch.max(feature_tensor, dim=2, keepdim=True)[0]  # [B, C, 1]
        x_avg = torch.mean(feature_tensor, dim=2, keepdim=True)  # [B, C, 1]

        x_max_feature = x_max.expand(-1, -1, N)  # [B, C, N] #N - number of points
        x_avg_feature = x_avg.expand(-1, -1, N)  # [B, C, N]

        x_global_feature = torch.cat(
            [x_max_feature, x_avg_feature], dim=1
        )  # [B, 2C, N]

        f_level_0 = self.propagation_0(
            x["points"].transpose(-1, -2),
            centers.transpose(-1, -2),
            x["points"].transpose(-1, -2),
            feature_tensor,
        )  # (B, N, D')

        x = torch.cat(
            (f_level_0, x_global_feature), dim=1
        )  # (B, N, D' + 2 * no. groups * transformer dim)

        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        x_hat = x.transpose(1, 2)  # (B, N, cls_dim)

        return x_hat

    def get_loss(self, x_hat, target):
        B, N, cls_dim = x_hat.shape
        x_hat = x_hat.reshape(B * N, cls_dim)
        target = target.reshape(B * N)

        return self.loss(x_hat, target)

    def get_miou(self, x_pred, target):
        """
        Compute mIoU following part segmentation convention:
        - IoU = 1.0 when class is absent in both pred and target
        - Average IoU per shape, then average across shapes
        """
        batch_size, num_points = x_pred.shape

        all_shape_ious = []

        for i in range(batch_size):
            pred_i = x_pred[i]
            target_i = target[i]

            part_ious = []
            # Check each possible class
            for class_id in range(self.cls_dim):
                gt_mask = target_i == class_id
                pred_mask = pred_i == class_id

                gt_count = gt_mask.sum()
                pred_count = pred_mask.sum()

                if gt_count == 0 and pred_count == 0:
                    # Class not present in both - perfect score
                    part_ious.append(torch.tensor(1.0, device=x_pred.device))
                elif gt_count == 0 or pred_count == 0:
                    # Class in one but not other - zero score
                    part_ious.append(torch.tensor(0.0, device=x_pred.device))
                else:
                    # Standard IoU calculation
                    intersection = (gt_mask & pred_mask).sum().float()
                    union = (gt_mask | pred_mask).sum().float()
                    iou = intersection / union
                    part_ious.append(iou)

            # Average IoU for this shape
            shape_iou = torch.stack(part_ious).mean()
            all_shape_ious.append(shape_iou)

        # Return mean IoU across all shapes
        return torch.stack(all_shape_ious).mean()

    def get_acc(self, x_pred, target):
        return self.accuracy(x_pred, target), self.balanced_accuracy(x_pred, target)

    def training_step(self, batch, batch_idx):
        x_hat = self.forward(batch)  # Logits (batch, N, cls_dim)
        x_gt = (
            batch["features"].squeeze(-1).long()
        )  # Ground truth points (batch, N) #Cls labels
        loss = self.get_loss(x_hat, x_gt)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )

        with torch.no_grad():
            x_pred = torch.argmax(x_hat, dim=2).long()  # Predicted classes (batch, N)
            acc, bal_acc = self.get_acc(x_pred, x_gt)
            self.log(
                "train/acc",
                acc,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "train/bal_acc",
                bal_acc,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            miou = self.get_miou(x_pred, x_gt)
            self.log(
                "train/miou",
                miou,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x_hat = self.forward(batch)  # Logits (batch, N, cls_dim)
        x_gt = (
            batch["features"].squeeze(-1).long()
        )  # Ground truth points (batch, N) #Cls labels
        x_pred = torch.argmax(x_hat, dim=2).long()  # Predicted classes (batch, N)

        # Debug prints
        # print("Unique predictions:", torch.unique(x_pred).cpu().numpy())
        # print("Unique ground truth:", torch.unique(x_gt).cpu().numpy())
        # print("Prediction distribution:", torch.bincount(x_pred.flatten()).cpu().numpy())
        # print("Ground truth distribution:", torch.bincount(x_gt.flatten()).cpu().numpy())

        loss = self.get_loss(x_hat, x_gt)
        acc, bal_acc = self.get_acc(x_pred, x_gt)
        miou = self.get_miou(x_pred, x_gt)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        self.log(
            "val/acc", acc, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        self.log(
            "val/bal_acc",
            bal_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/miou", miou, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x_hat = self.forward(batch)  # Logits (batch, N, cls_dim)
        x_gt = (
            batch["features"].squeeze(-1).long()
        )  # Ground truth points (batch, N) #Cls labels
        x_pred = torch.argmax(x_hat, dim=2).long()  # Predicted classes (batch, N)
        loss = self.get_loss(x_hat, x_gt)
        acc, bal_acc = self.get_acc(x_pred, x_gt)
        miou = self.get_miou(x_pred, x_gt)
        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        self.log(
            "test/acc", acc, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        self.log(
            "test/bal_acc",
            bal_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/miou", miou, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        return loss

    def on_fit_start(self):
        self.logger.experiment.log(self.hparams)

        logger.info(
            f"Trainable params before freezing: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )
        if self.freeze_encoder:
            logger.info("Freezing encoder")
            logger.info("Freezing patch encoder")
            for param in self.patch_encoder.parameters():
                param.requires_grad = False
            logger.info("Freezing pos encoder")
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            logger.info("Freezing transformer encoder")
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
        logger.info(
            f"Trainable params after freezing: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.05
        )
        # return optimizer
        warmup_epochs = self.warmup_epochs

        def lr_lambda(current_epoch):
            if warmup_epochs >= self.total_epochs:
                # Only warmup phase, no decay
                lr = float(current_epoch + 1) / float(max(1, warmup_epochs))
                return min(lr, 1.0)
            else:
                if current_epoch < warmup_epochs:
                    return float(current_epoch + 1) / float(warmup_epochs)
                else:
                    decay_epochs = self.total_epochs - warmup_epochs
                    return 0.5 * (
                        1
                        + math.cos(
                            math.pi * (current_epoch - warmup_epochs) / decay_epochs
                        )
                    )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        ]
