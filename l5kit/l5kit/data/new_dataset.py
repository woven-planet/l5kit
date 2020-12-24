import bisect
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import DefaultDict, List, Union

import numpy as np
from tqdm import tqdm

from l5kit.data.filter import (
    get_agents_slice_from_frames,
    get_frames_slice_from_scenes,
    get_tl_faces_slice_from_frames,
)
from l5kit.data.zarr_dataset import ChunkedDataset


class Dataset:
    PATH_KEY = "npz_name"  # required in each dataset

    def __init__(self, path: Union[Path, str]) -> None:
        self.path = Path(path)
        self.dataset_dict: DefaultDict[str, List[int]] = defaultdict(list)
        self.npz_names: List[str] = []

    def open(self) -> "Dataset":
        reader = csv.DictReader(open(str(self.path / "dataset.txt"), "r"))
        fieldnames = reader.fieldnames
        assert fieldnames is not None, "error reading fieldnames"
        assert fieldnames[0] == self.PATH_KEY, f"{self.PATH_KEY} should be the first field name"
        assert len(fieldnames) > 1, f"no fields other than {self.PATH_KEY} found"

        keys = fieldnames[1:]
        print(f"this dataset contains: {keys}")

        for row in reader:
            npz_name = row[self.PATH_KEY]
            assert npz_name.endswith(".npz")
            self.npz_names.append(npz_name)

            for k in keys:
                els_count = int(row[k])
                self.dataset_dict[k].append(els_count)

        # store diffs for each key instead of counts
        for key in self.dataset_dict:
            self.dataset_dict[key] = np.cumsum(self.dataset_dict[key])
            setattr(self, key, Array(self, key))  # register this array

        return self

    def load_npz(self, scene_idx: int) -> np.ndarray:  # lazy load here
        """
        Load an npz given its idx in the dataset

        Args:
            scene_idx (int): the npz index in the dataset

        Returns:
            np.ndarray:
        """
        name = self.npz_names[scene_idx]
        scene_path = self.path / name
        assert scene_path.is_file()
        data = np.load(scene_path)
        return data

    @lru_cache(maxsize=256)
    def get_data_by_npz_idx(self, name: str, scene_idx: int) -> np.ndarray:  # cache only arrays, not full npz
        data = self.load_npz(scene_idx)
        return data[name]

    @staticmethod
    def save_from_zarr(chunked: ChunkedDataset, path: Union[Path, str]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)

        dataset_meta_path = path / "dataset.txt"

        fieldnames = ["npz_name", "scenes", "frames", "agents", "tl_faces"]
        writer = csv.DictWriter(open(str(dataset_meta_path), "w"), fieldnames)
        writer.writeheader()

        for idx_scene in tqdm(range(len(chunked.scenes))):
            scene = chunked.scenes[idx_scene : idx_scene + 1]

            frames = chunked.frames[get_frames_slice_from_scenes(scene[0])]
            agents = chunked.agents[get_agents_slice_from_frames(*frames[[0, -1]])]
            tl_faces = chunked.tl_faces[get_tl_faces_slice_from_frames(*frames[[0, -1]])]

            npz_name = f"data_{idx_scene}.npz"
            np.savez_compressed(path / npz_name, scenes=scene, frames=frames, agents=agents, tl_faces=tl_faces)
            line = {
                "npz_name": f"data_{idx_scene}.npz",
                "scenes": len(scene),
                "frames": len(frames),
                "agents": len(agents),
                "tl_faces": len(tl_faces),
            }
            writer.writerow(line)


class Array:
    def __init__(self, dataset: Dataset, name: str):
        self.dataset = dataset
        self.cumsum = dataset.dataset_dict[name]
        self.name = name

    def bisect(self, index: int) -> int:
        data_idx = bisect.bisect_right(self.cumsum, index, hi=len(self.cumsum) - 1)
        return data_idx

    def get_relative_index(self, idx: int, data_idx: int) -> int:
        if data_idx == 0:
            return idx
        return idx - self.cumsum[data_idx - 1]

    def __len__(self) -> int:
        return self.cumsum[-1]

    def __getitem__(self, item: Union[int, slice, str]) -> np.ndarray:
        if isinstance(item, int):
            if item < 0:
                item = len(self) + item

            assert 0 <= item < len(self)

            data_index = self.bisect(item)
            data = self.dataset.get_data_by_npz_idx(self.name, data_index)
            return data[self.get_relative_index(item, data_index)]

        elif isinstance(item, slice):
            start_index = item.start if item.start is not None else 0
            if start_index < 0:
                start_index = len(self) + start_index

            stop_index = item.stop if item.stop is not None else len(self)
            if stop_index < 0:
                stop_index = len(self) + stop_index

            assert 0 <= start_index <= stop_index

            data_index_start = self.bisect(start_index)
            data_index_end = self.bisect(stop_index)

            data = []
            for data_index in range(data_index_start, data_index_end + 1):
                data.append(self.dataset.get_data_by_npz_idx(self.name, data_index))
            data = np.concatenate(data)

            start_index = self.get_relative_index(start_index, data_index_start)
            stop_index = self.get_relative_index(stop_index, data_index_start)
            return data[slice(start_index, stop_index, item.step)]

        elif isinstance(item, str):  # TODO this should not be called in general as it's super slow!
            data_index_start = 0
            data_index_end = len(self.cumsum) - 1

            data = []
            for data_index in range(data_index_start, data_index_end + 1):
                data.append(self.dataset.get_data_by_npz_idx(self.name, data_index))
            return np.concatenate(data)[item]

        else:
            raise TypeError
