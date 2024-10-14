from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np
from loguru import logger
from progressbar import progressbar as pbar


class TlsNormalizer:
    def __init__(
        self,
        dataset,
        params,
        n_samples=1000,  # Note - this is number of files, not number of points
        out_dtype=np.float32,
    ):
        self.dataset = dataset
        self.params = params
        self.n_samples = n_samples
        self.out_dtype = out_dtype

        self.num_channels = params["num_channels"]
        self.has_labels = params["has_labels"]

        self.hash = get_hash(str(params) + str(n_samples) + str(out_dtype))

        self.stats_file = os.path.join(
            self.dataset.base_folder, f"stats/stats_{self.hash}.pkl"
        )
        logger.info(f"Creating directory for stats_file {self.stats_file}")
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)

        self.mean = None
        self.std = None

    def prepare_data(self, force_compute=False):
        if not os.path.isfile(self.stats_file) or force_compute:
            if self.dataset.split != "train":
                raise ValueError(
                    "Normalizer can only be computed on the training set. Run prepare_data on the training set first."
                )

            logger.info(
                f"{self.dataset.__class__.__name__}.{self.params}.{self.dataset.split}.{self.hash}: computing normalizer"
            )

            x_sum = None
            xsq_sum = None

            n = 0
            failed_files = 0
            for idx in pbar(np.random.permutation(len(self.dataset))[: self.n_samples]):
                try:
                    x = self.dataset.load_item(idx)  # Might be 12414221x4 for example
                except Exception as e:
                    failed_files += 1
                    logger.warning(f"Failed to load item {idx}: {e}")
                    logger.warning(f"Failed files: {failed_files}")
                    continue

                if x_sum is None:
                    x_sum = np.zeros(x.shape[1])
                    xsq_sum = np.zeros(x.shape[1])

                x_sum += x.sum(axis=0)
                xsq_sum += (x**2).sum(axis=0)
                n += x.shape[0]  # Each pixel is one datapoint

            # Compute mean + std per channel
            self.mean = x_sum / n
            self.std = np.sqrt(xsq_sum / n - self.mean**2)

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

        num_remaining_channels = min(self.num_channels - 3 - int(self.has_labels), 0)

        self.mean = self.mean.astype(self.out_dtype)
        self.std = self.std.astype(self.out_dtype)

        logger.info(f"Dataset has {self.num_channels} channels, of which 3 are spatial, {num_remaining_channels} are non-spatial, and {int(self.has_labels)} are labels")
        logger.info(f"3 channels will be zero centered and scaled to [-1,1], {num_remaining_channels} channels will be normalized with mean and std")

    def normalize(self, x):
        # Zero mean and [-1,1] for the first 3 channels
        x[:, :3] = (x[:, :3] - self.mean) #Zero center
        max_val = np.abs(x[:, :3]).max() 
        x[:, :3] /= max_val

        #Other (non-spatial) channels are normalized with mean and std
        num_remaining_channels = x.shape[1] - 3

        if self.has_labels:
            num_remaining_channels -= 1 #Don't normalise the labels

        if num_remaining_channels > 0:
            x[:, 3:3 + num_remaining_channels] = (x[:, 3:3 + num_remaining_channels] - self.mean[3:3 + num_remaining_channels]) / self.std[3:3 + num_remaining_channels]

        return x.astype(self.out_dtype)


def check_file_exists(file: str) -> bool:
    """
    checks if a file exists, works for local files and files in a bucket
    """
    return os.path.isfile(file)


def check_dir_exists(dir: str) -> bool:
    """
    checks if a directory exists, works for local directories and directories in a bucket
    """
    return os.path.isdir(dir)


def list_all_files(dir: str) -> list:
    """
    returns a list of all files in a directory, works for local directories and directories in a bucket
    """
    return os.listdir(dir)


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
