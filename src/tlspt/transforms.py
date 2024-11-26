from __future__ import annotations

import numpy as np
from loguru import logger
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
                points.shape[0], min(self.num_points, points.shape[0]), replace=False
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
                idx = np.random.choice(points.shape[0], self.num_points, replace=False)

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


class HDF5ChunkSampler:
    """
    Uniformly samples a point cloud indices in blocks of size chunk_size (in row dir).
    Performed before retrieving the data from the HDF5 file.
    Returns row indices to be used in the HDF5 file.
    """

    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, chunk_size, length):
        if self.num_points % chunk_size != 0:  # Num samples doesn't divide in to chunks
            if self.num_points < chunk_size:
                logger.warning(
                    f"Number of points ({self.num_points}) is less than chunk size ({chunk_size}). Loading might be slow"
                )
            if self.num_points > chunk_size:
                logger.debug(
                    f"DEBUG: Number of points ({self.num_points}) is not divisible by chunk size ({chunk_size}). Loading might be slow"
                )

        if length <= self.num_points:
            return [(0, length)]

        num_chunks_ds = int(np.ceil(length / chunk_size))
        num_chunks_sample = int(np.ceil(self.num_points / chunk_size))

        all_chunk_indices = np.arange(num_chunks_ds)

        selected_chunk_indices = np.random.choice(
            all_chunk_indices, size=num_chunks_sample, replace=False
        )

        slice_starts = selected_chunk_indices * chunk_size
        slice_ends = np.minimum(slice_starts + chunk_size, length)

        total_points = np.sum(slice_ends - slice_starts)
        if (
            total_points < self.num_points
        ):  # Add more chunks if the final chunk is selected
            remaining_points = self.num_points - total_points
            remaining_chunks = np.setdiff1d(all_chunk_indices, selected_chunk_indices)
            remaining_chunk_indices = np.random.choice(
                remaining_chunks,
                size=int(np.ceil(remaining_points / chunk_size)),
                replace=False,
            )
            remaining_slice_starts = remaining_chunk_indices * chunk_size
            remaining_slice_ends = np.minimum(
                remaining_slice_starts + chunk_size, length
            )
            slice_starts = np.concatenate([slice_starts, remaining_slice_starts])
            slice_ends = np.concatenate([slice_ends, remaining_slice_ends])
            total_points = np.sum(slice_ends - slice_starts)

        if total_points > self.num_points:  # Trim the last chunk
            slice_ends[-1] = slice_ends[-1] - (total_points - self.num_points)

        assert (
            slice_ends[-1] - slice_starts[-1] <= chunk_size
            and slice_ends[-1] - slice_starts[-1] > 0
        )

        slices = [(start, end) for start, end in zip(slice_starts, slice_ends)]
        slices.sort(key=lambda x: x[0])

        # Merge adjacent slices
        i = 0
        while i < len(slices) - 1:
            if slices[i][1] == slices[i + 1][0]:
                slices[i] = (slices[i][0], slices[i + 1][1])
                slices.pop(i + 1)
            else:
                i += 1

        return slices


class Padder:
    """
    Pads rows with repeat values to the desired length
    """

    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, datapoint: dict) -> dict:
        points = datapoint["points"]
        features = datapoint["features"] if "features" in datapoint else None

        if points.shape[0] < self.num_points:
            points = points.tile((int(np.ceil(self.num_points / points.shape[0])), 1))[
                : self.num_points
            ]
            features = (
                features.tile((int(np.ceil(self.num_points / features.shape[0])), 1))[
                    : self.num_points
                ]
                if features is not None
                else None
            )

        datapoint["points"] = points
        if "features" in datapoint:
            datapoint["features"] = features

        return datapoint


class UniformTLSSampler:
    def __init__(self, num_points):
        self.num_points = num_points

        self.transform = Compose(
            [
                UniformDownsample(self.num_points, replace=False),  # Changes lengths
                Padder(self.num_points),  # Does not change lengths
            ]
        )

    def __call__(self, datapoint: dict) -> dict:
        return self.transform(datapoint)
