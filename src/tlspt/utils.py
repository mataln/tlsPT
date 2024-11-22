from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np
import torch
from loguru import logger
from progressbar import progressbar as pbar

from tlspt.structures.pointclouds import TLSPointclouds


class TlsNormalizer:
    def __init__(
        self,
        dataset,
        params=None,
        n_samples=1000,  # Note - this is number of files, not number of points
        out_dtype=torch.float,
    ):
        self.dataset = dataset
        self.params = params
        self.n_samples = n_samples
        self.out_dtype = out_dtype

        self.features_to_normalize = params.get("features_to_normalize", None)

        self.hash = get_hash(
            self.dataset.__repr__() + str(params) + str(n_samples) + str(out_dtype)
        )

        self.stats_file = os.path.join(
            self.dataset.base_folder, f"stats/stats_{self.hash}.pkl"
        )
        logger.info(f"Creating directory for stats_file {self.stats_file}")
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)

        self.mean = None
        self.std = None

    def prepare_data(self, force_compute=False):
        if self.dataset.feature_names is None:
            logger.info("No features to normalize")
            return
        if not os.path.isfile(self.stats_file) or force_compute:
            if self.dataset.split != "train":
                raise ValueError(
                    "Normalizer can only be computed on the training set. Run prepare_data on the training set first."
                )

            logger.info(
                f"{self.dataset.__class__.__name__}.{self.params}.{self.dataset.split}.{self.hash}: computing normalizer"
            )

            f_sum = None
            fsq_sum = None

            n = 0
            failed_files = 0
            for idx in pbar(
                np.random.permutation(len(self.dataset))[: self.n_samples]
            ):  # Over features
                try:
                    f = self.dataset.load_item(
                        idx
                    ).features_packed()  # Get features nxC
                except Exception as e:
                    failed_files += 1
                    logger.warning(f"Failed to load item {idx}: {e}")
                    logger.warning(f"Failed files: {failed_files}")
                    continue

                if f_sum is None:
                    f_sum = torch.zeros(f.shape[1])
                    fsq_sum = torch.zeros(f.shape[1])

                f_sum += f.sum(axis=0)
                fsq_sum += (f**2).sum(axis=0)
                n += f.shape[0]  # Each pixel is one datapoint

            # Compute mean + std per channel
            self.mean = f_sum / n
            self.std = torch.sqrt(fsq_sum / n - self.mean**2)

            # Save stats
            with open(self.stats_file, "wb") as f:
                pickle.dump({"mean": self.mean, "std": self.std}, f)

            logger.info(
                f"saved stats to {self.stats_file} with means {self.mean} and stds {self.std}"
            )

        logger.info(f"reading stats from {self.stats_file}")
        logger.info(
            f"for dataset {self.dataset.__class__.__name__}.{self.params}.{self.dataset.split}.{self.hash}"
        )

        with open(self.stats_file, "rb") as f:
            k = pickle.load(f)
            self.mean = k["mean"]
            self.std = k["std"]

        logger.info(f"mean: {self.mean}")
        logger.info(f"std: {self.std}")

        self.num_features = len(self.dataset.feature_names)

        logger.info(f"{self.out_dtype}")
        self.mean = self.mean.type(self.out_dtype)
        self.std = self.std.type(self.out_dtype)

        logger.info(
            f"Dataset has {len(self.dataset.feature_names)} features named {self.dataset.feature_names}. \n Normalizing {self.features_to_normalize} by mean+std. "
        )
        logger.info(f"3 channels will be zero centered and scaled to [-1,1].")

    def normalize(self, x):
        points = x.points_packed()
        features = x.features_packed()

        points = points - points.mean(dim=0)
        points /= points.abs().max()

        if self.dataset.feature_names is None:
            return TLSPointclouds(points=points.unsqueeze(0), features=None)

        if self.features_to_normalize is not None:
            idx_to_normalize = [
                self.dataset.feature_names.index(f) for f in self.features_to_normalize
            ]
            features[:, idx_to_normalize] = (
                features[:, idx_to_normalize] - self.mean[idx_to_normalize]
            ) / self.std[idx_to_normalize]

        return TLSPointclouds(
            points=points.unsqueeze(0),
            features=features.unsqueeze(0),
            feature_names=self.dataset.feature_names,
        )


def check_file_exists(file: str) -> bool:
    """
    checks if a file exists, works for local files and files in a bucket
    """
    return os.path.isfile(file)


def check_dir_exists(dir: str) -> bool:
    """
    checks if a directory exists, works for local directories
    """
    return os.path.isdir(dir)


def list_all_files(dir: str) -> list:
    """
    returns a list of all files in a directory, works for local directories
    """
    return os.listdir(dir)


def list_all_folders(dir: str) -> list:
    """
    returns a list of all folders in a directory, works for local directories
    """
    return [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]


def get_hash(s: str):
    """
    s: a string
    returns a hash string for s
    """
    if not isinstance(s, str):
        s = str(s)
    k = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k


class Hdf5Normalizer(TlsNormalizer):
    def __init__(
        self,
        dataset,
        params=None,
        n_samples=1000,  # Note - this is number of files, not number of points
        out_dtype=torch.float,
    ):
        super().__init__(dataset, params, n_samples, out_dtype)

    def prepare_data(self, force_compute=False):
        if self.dataset.feature_names is None:
            logger.info("No features to normalize")
            return
        if not os.path.isfile(self.stats_file) or force_compute:
            if self.dataset.split != "train":
                raise ValueError(
                    "Normalizer can only be computed on the training set. Run prepare_data on the training set first."
                )

            logger.info(
                f"{self.dataset.__class__.__name__}.{self.params}.{self.dataset.split}.{self.hash}: computing normalizer"
            )

            f_sum = None
            fsq_sum = None

            n = 0
            failed_files = 0
            for idx in pbar(
                np.random.permutation(len(self.dataset))[: self.n_samples]
            ):  # Over features
                try:
                    _, f = self.dataset.load_item(idx)  # Get features nxC
                except Exception as e:
                    failed_files += 1
                    logger.warning(f"Failed to load item {idx}: {e}")
                    logger.warning(f"Failed files: {failed_files}")
                    continue

                if f_sum is None:
                    f_sum = torch.zeros(f.shape[1])
                    fsq_sum = torch.zeros(f.shape[1])

                f_sum += f.sum(axis=0)
                fsq_sum += (f**2).sum(axis=0)
                n += f.shape[0]  # Each pixel is one datapoint

            # Compute mean + std per channel
            self.mean = f_sum / n
            self.std = torch.sqrt(fsq_sum / n - self.mean**2)

            # Save stats
            with open(self.stats_file, "wb") as f:
                pickle.dump({"mean": self.mean, "std": self.std}, f)

            logger.info(
                f"saved stats to {self.stats_file} with means {self.mean} and stds {self.std}"
            )

        logger.info(f"reading stats from {self.stats_file}")
        logger.info(
            f"for dataset {self.dataset.__class__.__name__}.{self.params}.{self.dataset.split}.{self.hash}"
        )

        with open(self.stats_file, "rb") as f:
            k = pickle.load(f)
            self.mean = k["mean"]
            self.std = k["std"]

        logger.info(f"mean: {self.mean}")
        logger.info(f"std: {self.std}")

        self.num_features = len(self.dataset.feature_names)

        logger.info(f"{self.out_dtype}")
        self.mean = self.mean.type(self.out_dtype)
        self.std = self.std.type(self.out_dtype)

        logger.info(
            f"Dataset has {len(self.dataset.feature_names)} features named {self.dataset.feature_names}. \n Normalizing {self.features_to_normalize} by mean+std. "
        )
        logger.info(f"3 channels will be zero centered and scaled to [-1,1].")

    def normalize(self, x, scale):
        points = x["points"]
        features = x["features"] if "features" in x else None

        points = points - points.mean(dim=0)  # Zero center
        points /= scale  # Scale by real world bbox size rather than pc

        if self.dataset.feature_names is None:
            return {"points": points, "lengths": x["lengths"]}

        if self.features_to_normalize is not None:
            idx_to_normalize = [
                self.dataset.feature_names.index(f) for f in self.features_to_normalize
            ]
            features[:, idx_to_normalize] = (
                features[:, idx_to_normalize] - self.mean[idx_to_normalize]
            ) / self.std[idx_to_normalize]

        return {"points": points, "features": features, "lengths": x["lengths"]}
