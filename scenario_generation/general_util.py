import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union



def get_dataset_path():
    cur_path = Path.cwd()
    while not (cur_path / "dataset_dir.txt").is_file():
        cur_path = cur_path.parent
    dataset_path = open(cur_path / "dataset_dir.txt", "r").read().strip()
    project_path = str(cur_path.resolve())
    return dataset_path, project_path