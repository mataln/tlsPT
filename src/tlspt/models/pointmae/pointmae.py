from __future__ import annotations

import lightning as L
from components import Group, MaskGenerator, PointNetEncoder, PositionEncoder
from torch import nn

from tlspt.models.transformer import TransformerEncoder
from tlspt.models.utils import get_unmasked


class PointMAE(L.LightningModule):
    def __init__(
        self,
        num_centers: int = X,
        num_neighbors: int = X,
        embedding_dim: int = X,
        mask_ratio: float = X,
        mask_type: str = "random",
        transformer_config: dict = X,
    ):
        super().__init__()
        self.num_centers = num_centers
        self.num_neighbors = num_neighbors
        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.transformer_config = transformer_config
        self.trans_dim = transformer_config["dim"]
        self.group = Group(
            num_centers=self.num_centers, num_neighbors=self.num_neighbors
        )
        self.pos_encoder = PositionEncoder(transformer_dim=self.trans_dim)
        self.patch_encoder = PointNetEncoder(embedding_dim=self.embedding_dim)
        self.mask_generator = MaskGenerator(
            mask_ratio=self.mask_ratio, mask_type=self.mask_type
        )
        self.transformer_encoder = (TransformerEncoder(**self.transformer_config),)
        self.norm = nn.LayerNorm(self.trans_dim)

    def forward_encoder(self, x):
        patches, centers = self.group(
            x
        )  # Get patches, patch centers. (batch, centers, neighbours, 3), (batch, centers, 3)

        mask = self.mask_generator(
            centers
        )  # Generate mask from centers (batch, centers)
        vis_centers = get_unmasked(
            centers, mask
        )  # Visible centers. (batch, centers, 3)
        vis_patches = get_unmasked(patches, mask)

        vis_patch_embeddings = self.patch_encoder(
            vis_patches
        )  # Embeddings for visible patches. (batch, centers, embedding_dim)
        vis_pos_embeddings = self.pos_encoder(
            vis_centers
        )  # Position embeddings for visible patches. (batch, centers, transformer_dim)

        # Feed visible patches and position embeddings to transformer
        x_vis = self.transformer_encoder(x_vis, vis_pos_embeddings)
        x_vis = self.norm(x_vis)

        return x_vis, mask

    def forward_decoder(self, x):
        pass

    def forward(self, x):
        x_vis, mask = self.forward_encoder(x)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
