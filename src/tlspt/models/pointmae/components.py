from __future__ import annotations

import torch
from pytorch3d.ops import knn_points, sample_farthest_points
from torch import nn
from torch.linalg import norm


class PointNetEncoder(nn.Module):
    """
    Encodes patches using mini pointnet
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, self.embedding_dim, 1),
        )

    def forward(self, x):
        """
        x: point groups; shape (batch size, no groups, no points per group, 3)
        out: embeddings; shape (batch size, no groups, embedding dim)
        """
        bs, g, n, _ = x.shape
        x = x.reshape(bs * g, n, 3)  # Each group is handled separately
        features = self.conv1(x.transpose(2, 1))
        global_features = torch.max(features, dim=2, keepdim=True)[0]
        features = torch.cat([global_features.expand(-1, -1, n), features], dim=1)
        features = self.conv2(features)
        global_features = torch.max(features, dim=2, keepdim=False)[0]
        return global_features.reshape(bs, g, self.embedding_dim)


class Group(nn.Module):
    """
    Generates patches with FPS + KNN
    """

    def __init__(self, num_centers, num_neighbors, normalize_group_distances=False):
        super().__init__()
        self.num_centers = num_centers
        self.num_neighbors = num_neighbors
        self.normalize_group_distances = normalize_group_distances

    def forward(self, x):
        """
        x: points; shape (batch size, no points, 3)
        num_centers: number of centers to sample using fps
        num_neighbors: number of neighbors to accrue for each center with knn
        """
        centers, center_idxs = sample_farthest_points(
            x, K=self.num_centers
        )  # (batch size, num centers, 3), (batch size, num centers)
        _, __, groups = knn_points(
            centers, x, K=self.num_neighbors, return_nn=True
        )  # (batch size, num centers, num neighbors, 3)

        groups = groups - centers.unsqueeze(2)  # Center about the centers
        if self.normalize_group_distances:
            max_distance = norm(groups, dim=-1).max(dim=-1, keepdim=True)[0]
            groups = groups / max_distance

        return groups, centers


class PositionEncoder(nn.Module):
    """
    Produces positional embeddings from point centers
    """

    def __init__(self, transformer_dim):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.transformer_dim),
        )

    def forward(self, x):
        """
        x: point centers; shape (batch size, no centers, 3)
        out: positional embeddings; shape (batch size, no centers, transformer dim)
        """
        return self.encoder(x)


class MaskGenerator(nn.Module):
    """
    Produces binary tensor mask for groups/patches
    """

    def __init__(self, mask_ratio, mask_type="random"):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def forward(self, x):
        """
        x: centers; shape (batch size, no groups, 3)
        out: mask; shape (batch size, no groups)
        """
        B, G, _ = x.shape

        if self.mask_ratio == 0:
            return torch.zeros(B, G, dtype=torch.bool)

        if self.mask_type == "random":
            num_masked = int(self.mask_ratio * G)
            mask = torch.zeros(B, G, dtype=torch.bool)
            for i in range(B):
                perm = torch.randperm(G)
                mask[i, perm[:num_masked]] = True

            return mask.to(x.device)

        raise ValueError(f"Invalid mask type: {self.mask_type}")
