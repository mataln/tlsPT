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


def get_at_index(x: torch.Tensor, idx: torch.Tensor):
    """
    Selects points from x according to mask indices
    x: (batch size, no centers, no neighbors, 3) OR (batch size, no centers, embedding dim)
    idx: (batch size, K)
    (Should also work on patch centers or embeddings)
    """
    device = x.device
    idx = idx.to(device)

    if x.dim() == 4:
        B, C, N, D = x.shape
        idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)
        return torch.gather(x, 1, idx_expanded).to(device).contiguous()
    elif x.dim() == 3:
        B, C, D = x.shape
        print(idx.shape)
        print(x.shape)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(x, 1, idx_expanded).to(device).contiguous()
    else:
        raise ValueError(f"Invalid dimension: {x.dim()}")


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points
