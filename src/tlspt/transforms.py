from __future__ import annotations

import numpy as np
from pytorch3d.ops import sample_farthest_points
from torchvision.transforms import Compose


class UniformDownsample:
    def __init__(self, num_points, replace="resample_as_req"):
        self.num_points = num_points
        self.replace = replace

    def __call__(self, datapoint: dict) -> dict:
        points = datapoint["points"]  # (n,3)
        if "features" in datapoint:
            features = datapoint["features"]  # (n,c)

        # Other cases
        if self.replace == False:  # Standard sampling
            idx = np.random.choice(
                np.arange(points.shape[0]), self.num_points, replace=False
            )
        elif self.replace == True:
            raise ValueError("Replacement not supported for uniform downsampling.")
        elif (
            self.replace == "resample_as_req"
        ):  # Replacement only if num_points > points.shape[0]
            if (
                self.num_points > points.shape[0]
            ):  # Note - order is NOT random. Prevents ball query from picking an identical point twice rather than 2 different ones where available.
                repeats = int(np.ceil(self.num_points / points.shape[0]))
                idx = np.tile(np.random.permutation(points.shape[0]), repeats)[
                    : self.num_points
                ]
            else:
                idx = np.random.choice(
                    np.arange(points.shape[0]), self.num_points, replace=False
                )

        datapoint["points"] = points[idx]
        if "features" in datapoint:
            datapoint["features"] = features[idx]

        datapoint["lengths"] = min(self.num_points, datapoint["lengths"])

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

        datapoint["lengths"] = min(self.num_points, datapoint["lengths"])

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

        raise NotImplementedError("Temporarily disabled.")

    def __call__(self, datapoint: dict) -> dict:
        return self.transform(datapoint)
