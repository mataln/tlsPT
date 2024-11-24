from __future__ import annotations

import lightning as L
import torch
from loguru import logger
from pytorch3d.loss import chamfer_distance
from torch import nn

from tlspt.models.transformer import TransformerEncoder
from tlspt.models.utils import get_at_index

from .components import Group, PointNetEncoder, PositionEncoder


class PointMAE(L.LightningModule):
    def __init__(
        self,
        backbone=None,
        num_centers: int = 64,
        num_neighbors: int = 32,
        embedding_dim: int = 384,
        neighbor_alg: str = "ball_query",
        ball_radius=None,
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
        total_epochs: int = 300,
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
            self.transencoder_config = backbone.transencoder_config  ##
            self.trans_dim = backbone.trans_dim  ##
            self.group = backbone.group
            self.pos_encoder = backbone.pos_encoder
            self.patch_encoder = backbone.patch_encoder
            self.transformer_encoder = backbone.transformer_encoder
            self.norm = backbone.norm
        else:
            self.num_centers = num_centers
            self.num_neighbors = num_neighbors
            self.embedding_dim = embedding_dim
            self.neighbor_alg = neighbor_alg
            self.ball_radius = ball_radius
            self.transencoder_config = transencoder_config
            self.trans_dim = transencoder_config["embed_dim"]
            self.group = Group(
                num_centers=self.num_centers,
                num_neighbors=self.num_neighbors,
                neighbor_alg=self.neighbor_alg,
                radius=self.ball_radius,
            )
            self.pos_encoder = PositionEncoder(transformer_dim=self.trans_dim)
            self.patch_encoder = PointNetEncoder(embedding_dim=self.embedding_dim)
            self.transformer_encoder = TransformerEncoder(**self.transencoder_config)
            self.norm = nn.LayerNorm(self.trans_dim)

        self.loss = XX
        self.total_epochs = total_epochs
        self.feature_blocks = feature_blocks

    def reconstruct(
        self, x
    ):  # Reconstructs points from decoded patch embeddings/tokens. x is (batch, no mask centers, transformer_dim)
        B, M, _ = x.shape
        x = x.transpose(1, 2)  # x is now (batch, transformer_dim, no mask centers)
        x = self.reconstructor(x)  # x is now (batch, 3*no neighbors, no mask centers)
        x = x.transpose(1, 2)  # x is now (batch, no mask centers, 3*no neighbors)
        x = x.reshape(B, M, -1, 3)  # x is now (batch, no mask centers, no neighbors, 3)
        return x

    def forward_encoder(self, patches, centers, unmasked_idx):
        patch_embeddings = self.patch_encoder(
            patches
        )  # Embeddings for visible patches. (batch, [1-m]*centers, embedding_dim)
        pos_embeddings = self.pos_encoder(
            centers
        )  # Position embeddings for visible patches. (batch, [1-m]*centers, transformer_dim)

        # Feed visible patches and position embeddings to transformer
        x, feature_tensor = self.transformer_encoder(
            patch_embeddings, pos_embeddings, feature_blocks=self.feature_blocks
        )
        x = self.norm(x)

        return x, pos_embeddings, feature_tensor

    def forward_decoder(self, x, full_pos_embeddings, N):
        x_rec = self.transformer_decoder(
            x, full_pos_embeddings, N
        )  # Reconstructed mask tokens (patch embeddings). (batch, no mask centers, transformer_dim)
        x_hat = self.reconstruct(
            x_rec
        )  # Reconstructed points. (batch, no mask centers, no neighbors, 3)

        return x_hat

    def forward(self, x):
        patches, centers = self.group(
            x["points"], x["lengths"]
        )  # patches (batch, no centers, no neighbors, 3), centers (batch, no centers, 3)

        # Encode
        x, pos_embeddings, feature_tensor = self.forward_encoder(
            patches, centers
        )  # x: (batch, centers, transformer_dim), pos_embeddings: (batch, centers, transformer_dim), feature_tensor: (no. blocks, B, no. centers, transformer dim)

        # Decode
        x_hat = self.forward_decoder(
            x_full, full_pos_embeddings, N
        )  # Decode to patch embeddings w/ missing patches, (batch, no mask centers, no neighbors, 3)
        x_gt = get_at_index(patches, masked_idx)

        loss = self.get_loss(x_hat, x_gt)
        return loss

    def get_loss(self, x_hat, target):
        B, M, N, _ = x_hat.shape
        x_hat = x_hat.reshape(B * M, N, 3)
        target = target.reshape(B * M, N, 3)

        if self.loss == chamfer_distance:
            loss = self.loss(x_hat, target)[0]
        else:
            raise NotImplementedError
        return loss

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.05)
        return optimizer
        # total_epochs = 300
        # warmup_epochs = 10

        # def lr_lambda(current_epoch):
        #     if current_epoch < warmup_epochs:
        #         return float(current_epoch) / float(max(1, warmup_epochs))
        #     else:
        #         return 0.5 * (1 + math.cos(math.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        # return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}]
