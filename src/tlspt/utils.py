from __future__ import annotations

import os


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
