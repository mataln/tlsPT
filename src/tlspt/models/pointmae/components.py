from __future__ import annotations

import torch
from pytorch3d.ops import ball_query, knn_points, sample_farthest_points
from torch import nn
from torch.linalg import norm

from tlspt.models.utils import get_at_index


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

    def __init__(
        self,
        num_centers,
        num_neighbors,
        neighbor_alg,
        radius=None,
        normalize_group_distances=False,
    ):
        super().__init__()
        self.num_centers = num_centers
        self.num_neighbors = num_neighbors
        self.normalize_group_distances = normalize_group_distances

        if neighbor_alg not in ["knn_points", "ball_query"]:
            raise ValueError(f"Invalid neighbor algorithm: {neighbor_alg}")
        self.neighbor_alg = neighbor_alg
        self.radius = radius
        if neighbor_alg == "ball_query" and radius is None:
            raise ValueError("Radius must be specified for ball query")

    def forward(self, points, lengths):
        """
        points: points; shape (batch size, no points, 3)
        lengths: number of points in each point cloud; shape (batch size)
        num_centers: number of centers to sample using fps
        num_neighbors: number of neighbors to accrue for each center with knn
        """
        centers, center_idxs = sample_farthest_points(
            points=points, lengths=lengths, K=self.num_centers
        )  # (batch size, num centers, 3), (batch size, num centers)
        if self.neighbor_alg == "knn_points":
            _, __, groups = knn_points(
                p1=centers,
                p2=points,
                lengths2=lengths,
                K=self.num_neighbors,
                return_nn=True,
            )  # (batch size, num centers, num neighbors, 3)
        elif self.neighbor_alg == "ball_query":
            _, __, groups = ball_query(
                p1=centers,
                p2=points,
                lengths2=lengths,
                K=self.num_neighbors,
                radius=self.radius,
                return_nn=True,
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


class MaskGeneratorBool(nn.Module):
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


class MaskGenerator(nn.Module):
    """
    Produces indices for masked points
    """

    def __init__(self, mask_ratio, mask_type="random"):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def forward(self, x):
        """
        x: centers; shape (batch size, no groups/centers, 3)
        out: mask; shape (batch size, mask_ratio * no groups) #e.g. mask[voxel 1] is a tensor of indices of masked points
        """
        B, G, _ = x.shape
        device = x.device

        if self.mask_ratio == 0:
            masked_idx = torch.empty(B, 0, dtype=torch.long, device=device)
            unmasked_idx = torch.arange(G, device=device).unsqueeze(0).expand(B, -1)
            return masked_idx, unmasked_idx

        if self.mask_type == "random":
            num_masked = int(self.mask_ratio * G)
            G - num_masked
            perm = torch.rand(B, G, device=device).argsort(dim=1)
            masked_idx = perm[:, :num_masked]
            unmasked_idx = perm[:, num_masked:]
            return masked_idx, unmasked_idx

        raise ValueError(f"Invalid mask type: {self.mask_type}")


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, N, D]
            points2: input points data, [B, S, D]
        Return:
            new_points: upsampled points data, [B, N, D']
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                get_at_index(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
