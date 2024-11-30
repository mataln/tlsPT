from __future__ import annotations

import lightning as L
import torch
from pytorch3d.loss import chamfer_distance
from torch import nn

from tlspt.models.transformer import TransformerDecoder, TransformerEncoder
from tlspt.models.utils import get_at_index

from .components import Group, MaskGenerator, PointNetEncoder, PositionEncoder


class PointMAE(L.LightningModule):
    def __init__(
        self,
        num_centers: int = 64,
        num_neighbors: int = 32,
        embedding_dim: int = 384,
        mask_ratio: float = 0.6,
        mask_type: str = "random",
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
        transdecoder_config: dict = {
            "embed_dim": 384,
            "depth": 4,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "qkv_bias": False,
            "qk_scale": None,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "norm_layer": nn.LayerNorm,
        },
        total_epochs: int = 300,
    ):
        raise NotImplementedError("You've run the wrong model")
        super().__init__()
        self.num_centers = num_centers
        self.num_neighbors = num_neighbors
        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.neighbor_alg = neighbor_alg
        self.ball_radius = ball_radius
        self.transencoder_config = transencoder_config
        self.trans_dim = transencoder_config["embed_dim"]
        if transdecoder_config["embed_dim"] != transencoder_config["embed_dim"]:
            raise ValueError("Encoder and decoder dimensions must match")
        self.transdecoder_config = transdecoder_config
        self.group = Group(
            num_centers=self.num_centers,
            num_neighbors=self.num_neighbors,
            neighbor_alg=self.neighbor_alg,
            radius=self.ball_radius,
        )
        self.pos_encoder = PositionEncoder(transformer_dim=self.trans_dim)
        self.patch_encoder = PointNetEncoder(embedding_dim=self.embedding_dim)
        self.mask_generator = MaskGenerator(
            mask_ratio=self.mask_ratio, mask_type=self.mask_type
        )
        self.transformer_encoder = TransformerEncoder(**self.transencoder_config)
        self.norm = nn.LayerNorm(self.trans_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.transformer_decoder = TransformerDecoder(**transdecoder_config)

        self.reconstructor = nn.Conv1d(
            self.trans_dim, 3 * self.num_neighbors, 1
        )  # Num neighbors points per group.
        self.loss = chamfer_distance
        self.total_epochs = total_epochs

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
        vis_centers = get_at_index(
            centers, unmasked_idx
        )  # Visible centers. (batch, [1-m]*centers, 3)
        vis_patches = get_at_index(
            patches, unmasked_idx
        )  # Visible patches. (batch, [1-m]*centers, neighbors, 3)

        vis_patch_embeddings = self.patch_encoder(
            vis_patches
        )  # Embeddings for visible patches. (batch, [1-m]*centers, embedding_dim)
        vis_pos_embeddings = self.pos_encoder(
            vis_centers
        )  # Position embeddings for visible patches. (batch, [1-m]*centers, transformer_dim)

        # Feed visible patches and position embeddings to transformer
        x_vis = self.transformer_encoder(vis_patch_embeddings, vis_pos_embeddings)
        x_vis = self.norm(x_vis)

        return x_vis, vis_pos_embeddings

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

        masked_idx, unmasked_idx = self.mask_generator(
            centers
        )  # Generate mask from centers (batch, centers)

        # Encode visible
        x_vis, vis_pos_embeddings = self.forward_encoder(
            patches, centers, unmasked_idx
        )  # x_vis: (batch, centers, transformer_dim), mask: (batch, centers)

        masked_centers = get_at_index(
            centers, masked_idx
        )  # Masked centers. (batch, m*centers, 3)
        masked_pos_embeddings = self.pos_encoder(
            masked_centers
        )  # batch, m*centers, transformer_dim

        B, N, _ = masked_pos_embeddings.shape
        mask_tokens = self.mask_token.expand(B, N, -1)

        x_full = torch.cat((x_vis, mask_tokens), dim=1)
        full_pos_embeddings = torch.cat(
            (vis_pos_embeddings, masked_pos_embeddings), dim=1
        )

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
