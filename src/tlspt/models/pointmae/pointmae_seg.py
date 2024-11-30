from __future__ import annotations

import math

import lightning as L
import torch
from loguru import logger
from torch import nn

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
        prop_mlp_dim: int = 1024,
    ):
        super().__init__()
        if backbone is not None:  # Preload
            logger.info(
                "Loading model from pretrained backbone, other model parameters will be ignored"
            )
            self.num_centers = backbone.num_centers
            self.num_neighbors = backbone.num_neighbors
            self.embedding_dim = backbone.embedding_dim
            self.neighbor_alg = backbone.neighbor_alg
            self.ball_radius = backbone.ball_radius
            self.scale = backbone.scale
            self.transencoder_config = backbone.transencoder_config  ##
            self.trans_dim = backbone.trans_dim  ##
            self.group = backbone.group
            self.pos_encoder = backbone.pos_encoder
            self.patch_encoder = backbone.patch_encoder
            self.transformer_encoder = backbone.transformer_encoder
            self.norm = backbone.norm
        else:
            if scale is None:
                raise ValueError(
                    "Scale must be provided when not using a pretrained model"
                )
            self.num_centers = num_centers
            self.num_neighbors = num_neighbors
            self.embedding_dim = embedding_dim
            self.neighbor_alg = neighbor_alg
            self.ball_radius = ball_radius
            self.scale = scale
            self.transencoder_config = transencoder_config
            self.trans_dim = transencoder_config["embed_dim"]
            self.group = Group(
                num_centers=self.num_centers,
                num_neighbors=self.num_neighbors,
                neighbor_alg=self.neighbor_alg,
                radius=self.ball_radius / self.scale,
            )
            self.pos_encoder = PositionEncoder(transformer_dim=self.trans_dim)
            self.patch_encoder = PointNetEncoder(embedding_dim=self.embedding_dim)
            self.transformer_encoder = TransformerEncoder(**self.transencoder_config)
            self.norm = nn.LayerNorm(self.trans_dim)

        self.loss = nn.CrossEntropyLoss()
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
        x = self.norm(x)

        return x, pos_embeddings, feature_list

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

        x_gt = x["features"]  # Ground truth points (batch, N, 1) #Cls labels
        if x["features"].shape[2] > 1:
            raise ValueError(
                "Multiple features not supported. Need to use label as only feature."
            )
        x_gt = x_gt.squeeze(-1)  # (batch, N)

        patches, centers = self.group(
            x["points"], x["lengths"]
        )  # patches (batch, no centers, no neighbors, 3), centers (batch, no centers, 3)

        # Encode
        _, pos_embeddings, feature_list = self.forward_encoder(
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

        loss = self.get_loss(x_hat, x_gt.long())
        return loss

    def get_loss(self, x_hat, target):
        B, N, cls_dim = x_hat.shape
        x_hat = x_hat.reshape(B * N, cls_dim)
        target = target.reshape(B * N)

        return self.loss(x_hat, target)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        pass

    def on_fit_start(self):
        self.logger.experiment.log(self.hparams)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.05)
        # return optimizer
        warmup_epochs = 10

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))
            else:
                return 0.5 * (
                    1
                    + math.cos(
                        math.pi
                        * (current_epoch - warmup_epochs)
                        / (self.total_epochs - warmup_epochs)
                    )
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        ]
