from __future__ import annotations

import torch


def get_masked(x: torch.Tensor, idx: torch.Tensor):
    """
    Selects points from x according to mask indices
    x: (batch size, no centers, no neighbors, 3) OR (batch size, no centers, embedding dim)
    idx: (batch size, no centers)
    (Should also work on patch centers or embeddings)
    """
    if x.dim() == 4:
        B, C, N, D = x.shape
        return x[idx].reshape(B, -1, N, D)
    elif x.dim() == 3:
        B, C, D = x.shape
        return x[idx].reshape(B, -1, D)


def get_unmasked(x: torch.Tensor, idx: torch.Tensor):
    """
    Selects points from x according to mask indices
    x: (batch size, no centers, no neighbors, 3) OR (batch size, no centers, embedding dim)
    idx: (batch size, no centers)
    (Should also work on patch centers or embeddings)
    """
    if x.dim() == 4:
        B, C, N, D = x.shape
        return x[~idx].reshape(B, -1, N, D)
    elif x.dim() == 3:
        B, C, D = x.shape
        return x[~idx].reshape(B, -1, D)
