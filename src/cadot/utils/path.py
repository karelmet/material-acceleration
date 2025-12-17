import os
import sys
from pathlib import Path

def get_src_path():
    """
    Return the path to the 'src' folder of the project.
    Assumes this file is located inside: project/src/cadot/utils/path.py
    """
    return Path(__file__).resolve().parents[2]


def get_project_root():
    """
    Return the root directory of the project.
    Equivalent to going one level above 'src'.
    """
    return get_src_path().parent


def get_data_path(data_dir_name):
    """
    Return the path to the dataset folder, e.g. project/DataCadot_yolo.
    You can change the folder name if needed.
    """
    return get_project_root() / data_dir_name


def add_src_to_sys_path():
    """
    Add the src directory to sys.path (useful in notebooks).
    Does nothing if it is already present.
    """
    src_path = str(get_src_path())
    if src_path not in sys.path:
        sys.path.append(src_path)