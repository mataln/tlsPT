from __future__ import annotations

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.embedding_dim, 1),
        )

    def forward(self, x):
        """
        x: point groups; shape (batch size, no groups, no points, 3)
        out: embeddings; shape (batch size, no groups, embedding dim)
        """
        bs, g, n, _ = x.shape
        x = x.rehsape(bs * g, n, 3)
        features = self.conv1(x.transpose(2, 1))
        global_features = torch.max(features, dim=2, keepdim=True)[0]
        features = torch.cat([global_features.expand(-1, -1, n), features], dim=1)
        features = self.conv2(features)
        global_features = torch.max(features, dim=2, keepdim=False)[0]
        return global_features.reshape(bs, g, self.embedding_dim)
