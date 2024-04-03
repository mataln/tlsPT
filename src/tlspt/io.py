from __future__ import annotations

import numpy as np

from tlspt import utils


def load_numpy(file_path: str, out_dtype: np.dtype = np.float32):
    """
    Loads a numpy file from disk
    """
    if not file_path.endswith(".npy"):
        raise ValueError(f"file {file_path} is not a numpy file")

    if not utils.check_file_exists(file_path):
        raise ValueError(f"cannot find file at {file_path}")

    arr = np.load(file_path)
    arr = arr.astype(out_dtype)
    return arr
