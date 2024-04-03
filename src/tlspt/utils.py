from __future__ import annotations

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
        force_compute=False,
    ):
        self.dataset = dataset
        self.params = params
        self.n_samples = n_samples
        self.out_dtype = (out_dtype,)
        self.force_compute = force_compute

        self.hash = get_hash(str(params))

        self.stats_file = f"{self.dataset.base_folder}/stats/stats_{self.hash}.pkl"
        logger.info(f"Creating directory for stats_file {self.stats_file}")
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)

    def prepare_data(self):
        if not os.path.isfile(self.stats_file) or self.force_compute:
            if self.dataset.split != "train":
                raise ValueError(
                    "Normalizer can only be computed on the training set. Run prepare_data on the training set first."
                )

            logger.info(
                f"{self.dataset.__class__.__name__}.{self.params}.{self.dataset.split}.{self.hash}: computing normalizer"
            )

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

        with pickle.load(open(self.stats_file, "rb")) as f:
            self.mean = f["mean"]
            self.std = f["std"]

        self.mean = self.mean.astype(self.out_dtype)
        self.std = self.std.astype(self.out_dtype)

    def normalize(self, x):
        # Placeholder
        # Currently normalizes all columns (except last) by scaling rather than z score
        min_vals = np.min(x[:, :-1], axis=0)
        max_vals = np.max(x[:, :-1], axis=0)

        x[:, :-1] = (x[:, :-1] - min_vals) / (max_vals - min_vals)  # Scaling

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
