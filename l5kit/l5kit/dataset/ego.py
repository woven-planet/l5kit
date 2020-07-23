import bisect
from functools import partial
from typing import Optional, Tuple, cast

import numpy as np
from torch.utils.data import Dataset

from ..data import ChunkedStateDataset
from ..kinematic import Perturbation
from ..rasterization import Rasterizer
from ..sampling import generate_agent_sample


class EgoDataset(Dataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedStateDataset,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNN

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedStateDataset): the raw zarr dataset
            rasterizer (Rasterizer): an object that support rasterisation around an agent (AV or not)
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
None if not desired
        """
        self.perturbation = perturbation
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.rasterizer = rasterizer

        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]

        # build a partial so we don't have to access cfg each time
        self.sample_function = partial(
            generate_agent_sample,
            raster_size=cast(Tuple[int, int], tuple(cfg["raster_params"]["raster_size"])),
            pixel_size=np.array(cfg["raster_params"]["pixel_size"]),
            ego_center=np.array(cfg["raster_params"]["ego_center"]),
            history_num_frames=cfg["model_params"]["history_num_frames"],
            history_step_size=cfg["model_params"]["history_step_size"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            future_step_size=cfg["model_params"]["future_step_size"],
            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=rasterizer,
            perturbation=perturbation,
        )

    def __len__(self) -> int:
        """
        Get the number of available AV frames

        Returns:
            int: the number of elements in the dataset
        """
        return len(self.dataset.frames)

    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
            the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp

        """
        frame_interval = self.dataset.scenes[scene_index]["frame_index_interval"]
        frames = self.dataset.frames[frame_interval[0] : frame_interval[1]]
        data = self.sample_function(state_index, frames, self.dataset.agents, self.dataset.tr_faces, track_id)
        # 0,1,C -> C,0,1
        image = data["image"].transpose(2, 0, 1)

        target_positions = np.array(data["target_positions"], dtype=np.float32)
        target_yaws = np.array(data["target_yaws"], dtype=np.float32)

        timestamp = self.dataset.frames[frame_interval[0] + state_index]["timestamp"]
        track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        return {
            "image": image,
            "target_positions": target_positions,
            "target_yaws": target_yaws,
            "target_availabilities": data["target_availabilities"],
            "world_to_image": data["world_to_image"],
            "track_id": track_id,
            "timestamp": timestamp,
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "extent": data["extent"],
        }

    def __getitem__(self, index: int) -> dict:
        """
        Function called by Torch to get an element

        Args:
            index (int): index of the element to retrieve

        Returns: please look get_frame signature and docstring

        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        scene_index = bisect.bisect_right(self.cumulative_sizes, index)

        if scene_index == 0:
            state_index = index
        else:
            state_index = index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index)

    def get_scene_dataset(self, scene_index: int) -> "EgoDataset":
        """
        Returns another EgoDataset dataset where the underlying data can be modified.
        This is possible because, even if it supports the same interface, this dataset is np.ndarray based.

        Args:
            scene_index (int): the scene index of the new dataset

        Returns:
            EgoDataset: A valid EgoDataset dataset with a copy of the data

        """
        # copy everything to avoid references (scene is already detached from zarr if get_combined_scene was called)
        scenes = self.dataset.scenes[scene_index : scene_index + 1].copy()
        frame_interval = scenes[0]["frame_index_interval"]
        frames = self.dataset.frames[frame_interval[0] : frame_interval[1]].copy()
        # ASSUMPTION: all agents_index are consecutive
        start_index = frames[0]["agent_index_interval"][0]
        end_index = frames[-1]["agent_index_interval"][1]
        agents = self.dataset.agents[start_index:end_index].copy()
        frames["agent_index_interval"] -= start_index
        scenes["frame_index_interval"] -= frame_interval[0]

        dataset = ChunkedStateDataset("")
        dataset.frames = frames
        dataset.agents = agents
        dataset.scenes = scenes

        return EgoDataset(self.cfg, dataset, self.rasterizer, self.perturbation)

    def get_scene_indices(self, scene_idx: int) -> np.ndarray:
        """
        Get indices for the given scene. EgoDataset iterates over frames, so this is just a matter
        of finding the scene boundaries.
        Args:
            scene_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        scenes = self.dataset.scenes
        assert scene_idx < len(scenes), f"scene_idx {scene_idx} is over len {len(scenes)}"
        return np.arange(*scenes[scene_idx]["frame_index_interval"])

    def get_frame_indices(self, frame_idx: int) -> np.ndarray:
        """
        Get indices for the given frame. EgoDataset iterates over frames, so this will be a single element
        Args:
            frame_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        frames = self.dataset.frames
        assert frame_idx < len(frames), f"frame_idx {frame_idx} is over len {len(frames)}"
        return np.asarray((frame_idx,), dtype=np.int64)

    def __repr__(self) -> str:
        return self.dataset.__repr__()
