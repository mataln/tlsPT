from __future__ import annotations

import numpy as np
from pytorch3d.ops import sample_farthest_points
from torchvision.transforms import Compose


class UniformDownsample:
    def __init__(self, num_points, replace=False):
        self.num_points = num_points
        self.replace = replace

    def __call__(self, datapoint: dict) -> dict:
        points = datapoint["points"]
        if "features" in datapoint:
            features = datapoint["features"]

        if self.replace != "as_req":
            idx = np.random.choice(
                np.arange(points.shape[0]), self.num_points, replace=self.replace
            )
        else:
            if self.num_points > points.shape[0]:
                idx = np.random.choice(
                    np.arange(points.shape[0]), self.num_points, replace=True
                )
            else:
                idx = np.random.choice(
                    np.arange(points.shape[0]), self.num_points, replace=False
                )

        datapoint["points"] = points[idx]
        if "features" in datapoint:
            datapoint["features"] = features[idx]

        return datapoint


class FarthestPointSample:
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, datapoint: dict) -> dict:
        points = datapoint["points"].unsqueeze(0)
        if "features" in datapoint:
            features = datapoint["features"]

        datapoint["points"], idx = sample_farthest_points(points, K=self.num_points)
        datapoint["points"] = datapoint["points"].squeeze(0)
        idx = idx.squeeze(0)
        if "features" in datapoint:
            datapoint["features"] = features[idx]

        return datapoint


class TLSSampler:
    def __init__(self, uniform_points, farthest_points):
        self.uniform = uniform_points
        self.farthest = farthest_points

        self.transform = Compose(
            [
                UniformDownsample(self.uniform, replace="as_req"),
                FarthestPointSample(self.farthest),
            ]
        )

    def __call__(self, datapoint: dict) -> dict:
        return self.transform(datapoint)
