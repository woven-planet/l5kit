import bisect
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset
from zarr import convenience

from l5kit.data import ChunkedDataset, get_agents_slice_from_frames, get_frames_slice_from_scenes
from l5kit.data.labels import PERCEPTION_LABEL_TO_INDEX
from l5kit.dataset.utils import convert_str_to_fixed_length_tensor
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer, RenderContext
from l5kit.sampling.agent_sampling import generate_agent_sample
from l5kit.sampling.agent_sampling_vectorized import generate_agent_sample_vectorized
from l5kit.vectorization.vectorizer import Vectorizer

from .select_agents import select_agents, TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE
from .utils import AGENT_MIN_FRAME_FUTURE, AGENT_MIN_FRAME_HISTORY


class BaseEgoDataset(Dataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNN

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
        """
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]

        # build a partial so we don't have to access cfg each time
        self.sample_function = self._get_sample_function()

    def _get_sample_function(self) -> Callable[..., dict]:
        raise NotImplementedError()

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
            dict: the rasterised image in (Cx0x1) if the rast is not None, the target trajectory
            (position and yaw) along with their availability, the 2D matrix to center that agent,
            the agent track (-1 if ego) and the timestamp

        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

        tl_faces = self.dataset.tl_faces
        # TODO (@lberg): this should be done in the sample function
        if self.cfg["raster_params"]["disable_traffic_light_faces"]:
            tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces

        data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)

        # add information only, so that all data keys are always preserved
        data["scene_index"] = scene_index
        data["host_id"] = np.uint8(convert_str_to_fixed_length_tensor(self.dataset.scenes[scene_index]["host"]).cpu())
        data["timestamp"] = frames[state_index]["timestamp"]
        data["track_id"] = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        return data

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

    def get_scene_dataset(self, scene_index: int) -> "BaseEgoDataset":
        """
        Returns another EgoDataset dataset where the underlying data can be modified.
        This is possible because, even if it supports the same interface, this dataset is np.ndarray based.

        Args:
            scene_index (int): the scene index of the new dataset

        Returns:
            EgoDataset: A valid EgoDataset dataset with a copy of the data

        """
        dataset = self.dataset.get_scene_dataset(scene_index)
        return BaseEgoDataset(self.cfg, dataset)

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

    def __str__(self) -> str:
        return self.dataset.__str__()


class EgoDataset(BaseEgoDataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNN

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            rasterizer (Rasterizer): an object that support rasterisation around an agent (AV or not)
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
            None if not desired
        """
        self.perturbation = perturbation
        self.rasterizer = rasterizer
        super().__init__(cfg, zarr_dataset)

    def _get_sample_function(self) -> Callable[..., dict]:
        render_context = RenderContext(
            raster_size_px=np.array(self.cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(self.cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(self.cfg["raster_params"]["ego_center"]),
            set_origin_to_bottom=self.cfg["raster_params"]["set_origin_to_bottom"],
        )

        return partial(
            generate_agent_sample,
            render_context=render_context,
            history_num_frames=self.cfg["model_params"]["history_num_frames"],
            future_num_frames=self.cfg["model_params"]["future_num_frames"],
            step_time=self.cfg["model_params"]["step_time"],
            filter_agents_threshold=self.cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=self.rasterizer,
            perturbation=self.perturbation,
        )

    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        data = super().get_frame(scene_index, state_index, track_id=track_id)
        # TODO (@lberg): this should not be here but in the rasterizer
        data["image"] = data["image"].transpose(2, 0, 1)  # 0,1,C -> C,0,1
        return data

    def get_scene_dataset(self, scene_index: int) -> "EgoDataset":
        """
        Returns another EgoDataset dataset where the underlying data can be modified.
        This is possible because, even if it supports the same interface, this dataset is np.ndarray based.

        Args:
            scene_index (int): the scene index of the new dataset

        Returns:
            EgoDataset: A valid EgoDataset dataset with a copy of the data

        """
        dataset = self.dataset.get_scene_dataset(scene_index)
        return EgoDataset(self.cfg, dataset, self.rasterizer, self.perturbation)


class EgoDatasetVectorized(BaseEgoDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        vectorizer: Vectorizer,
        perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNNs with vectorized input

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            vectorizer (Vectorizer): a object that supports vectorization around an AV
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
        None if not desired
        """
        self.perturbation = perturbation
        self.vectorizer = vectorizer
        super().__init__(cfg, zarr_dataset)

    def _get_sample_function(self) -> Callable[..., dict]:
        return partial(
            generate_agent_sample_vectorized,
            history_num_frames_ego=self.cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=self.cfg["model_params"]["history_num_frames_agents"],
            future_num_frames=self.cfg["model_params"]["future_num_frames"],
            step_time=self.cfg["model_params"]["step_time"],
            filter_agents_threshold=self.cfg["raster_params"]["filter_agents_threshold"],
            perturbation=self.perturbation,
            vectorizer=self.vectorizer
        )

    def get_scene_dataset(self, scene_index: int) -> "EgoDatasetVectorized":
        dataset = self.dataset.get_scene_dataset(scene_index)
        return EgoDatasetVectorized(self.cfg, dataset, self.vectorizer, self.perturbation)


class EgoAgentDatasetVectorized(EgoDatasetVectorized):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        vectorizer: Vectorizer,
        perturbation: Optional[Perturbation] = None,
        agents_mask: Optional[np.ndarray] = None,
        eval_mode: bool = False
    ):
        """
        Get a PyTorch dataset object that can be used to train DNNs with vectorized input.
        Ego features are added to the agent features to treat ego as an agent.

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            vectorizer (Vectorizer): a object that supports vectorization around an AV
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
                None if not desired
            agents_mask (Optional[np.ndarray]): custom boolean mask of the agent availability.
            eval_mode (bool): enable eval mode (iterates over agent, similarly to AgentDataset).
        """
        super().__init__(cfg, zarr_dataset, vectorizer, perturbation)

        self.eval_mode = eval_mode
        self.vectorizer.other_agents_num -= 1  # account for ego as additional agent

        min_frame_history = AGENT_MIN_FRAME_HISTORY
        min_frame_future = AGENT_MIN_FRAME_FUTURE
        if agents_mask is None:  # if not provided try to load it from the zarr
            agents_mask = self.load_agents_mask()
            past_mask = agents_mask[:, 0] >= min_frame_history
            future_mask = agents_mask[:, 1] >= min_frame_future
            agents_mask = past_mask * future_mask

            if min_frame_history != AGENT_MIN_FRAME_HISTORY:
                warnings.warn(
                    f"you're running with custom min_frame_history of {min_frame_history}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if min_frame_future != AGENT_MIN_FRAME_FUTURE:
                warnings.warn(
                    f"you're running with custom min_frame_future of {min_frame_future}", RuntimeWarning, stacklevel=2
                )
        else:
            warnings.warn("you're running with a custom agents_mask", RuntimeWarning, stacklevel=2)

        # store the valid agents indices (N_valid_agents,)
        self.agents_indices = np.nonzero(agents_mask)[0]

        # store an array where valid indices have increasing numbers and the rest is -1 (N_total_agents,)
        self.mask_indices = agents_mask.copy().astype(np.int)
        self.mask_indices[self.mask_indices == 0] = -1
        self.mask_indices[self.mask_indices == 1] = np.arange(0, np.sum(agents_mask))

        # this will be used to get the frame idx from the agent idx
        self.cumulative_sizes_agents = self.dataset.frames["agent_index_interval"][:, 1]
        self.agents_mask = agents_mask

    def __len__(self) -> int:
        """
        Get the number of available AV frames if not eval mode, otherwise return the number of available scenes

        Returns:
            int: the number of elements in the dataset
        """
        return len(self.dataset.frames) if not self.eval_mode else len(self.dataset.scenes)

    def load_agents_mask(self) -> np.ndarray:
        """
        Loads a boolean mask of the agent availability stored into the zarr. Performs some sanity check against cfg.
        Returns: a boolean mask of the same length of the dataset agents

        """
        agent_prob = self.cfg["raster_params"]["filter_agents_threshold"]

        agents_mask_path = Path(self.dataset.path) / f"agents_mask/{agent_prob}"
        if not agents_mask_path.exists():  # don't check in root but check for the path
            warnings.warn(
                f"cannot find the right config in {self.dataset.path},\n"
                f"your cfg has loaded filter_agents_threshold={agent_prob};\n"
                "but that value doesn't have a match among the agents_mask in the zarr\n"
                "Mask will now be generated for that parameter.",
                RuntimeWarning,
                stacklevel=2,
            )

            select_agents(
                self.dataset,
                agent_prob,
                th_yaw_degree=TH_YAW_DEGREE,
                th_extent_ratio=TH_EXTENT_RATIO,
                th_distance_av=TH_DISTANCE_AV,
            )

        agents_mask = convenience.load(str(agents_mask_path))  # note (lberg): this doesn't update root
        return agents_mask

    def get_scene_dataset(self, scene_index: int) -> "EgoAgentDatasetVectorized":
        """
        Returns another EgoAgentDatasetVectorized dataset where the underlying data can be modified.
        This is possible because, even if it supports the same interface, this dataset is np.ndarray based.

        Args:
            scene_index (int): the scene index of the new dataset

        Returns:
            EgoAgentDatasetVectorized: A valid EgoAgentDatasetVectorized dataset with a copy of the data

        """
        dataset = self.dataset.get_scene_dataset(scene_index)
        agents_mask = self.agents_mask

        # filter agents_bool values
        frame_interval = self.dataset.scenes[scene_index]["frame_index_interval"]
        # ASSUMPTION: all agents_index are consecutive
        start_index = self.dataset.frames[frame_interval[0]]["agent_index_interval"][0]
        end_index = self.dataset.frames[frame_interval[1] - 1]["agent_index_interval"][1]
        agents_mask = agents_mask[start_index:end_index].copy()

        return EgoAgentDatasetVectorized(self.cfg, dataset, self.vectorizer, self.perturbation, agents_mask)

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
        frame_slice = get_frames_slice_from_scenes(scenes[scene_idx])
        agent_slice = get_agents_slice_from_frames(*self.dataset.frames[frame_slice][[0, -1]])

        mask_valid_indices = (self.agents_indices >= agent_slice.start) * (self.agents_indices < agent_slice.stop)
        indices = np.nonzero(mask_valid_indices)[0]
        return indices

    def get_frame_indices(self, frame_idx: int) -> np.ndarray:
        """
        Get indices for the given frame. EgoDataset iterates over frames, so this will be a single element
        Args:
            frame_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        # avoid using `get_agents_slice_from_frames` as it hits the disk
        agent_start = self.cumulative_sizes_agents[frame_idx - 1] if frame_idx > 0 else 0
        agent_end = self.cumulative_sizes_agents[frame_idx]
        # slice using frame boundaries and take only valid indices
        mask_idx = self.mask_indices[agent_start:agent_end]
        indices = mask_idx[mask_idx != -1]
        return indices

    def __getitem__(self, index: int) -> dict:
        """
        Function called by Torch to get an element.
        Extends the output of the EgoDatasetVectorized to concatenate ego and agents as agent features.

        Args:
            index (int): index of the element to retrieve

        Returns: please look get_frame signature and docstring
        """
        # data_batch = super().__getitem__(index)
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        # get the scene
        if not self.eval_mode:
            scene_index = bisect.bisect_right(self.cumulative_sizes, index)
            if scene_index == 0:
                state_index = index
            else:
                state_index = index - self.cumulative_sizes[scene_index - 1]
            data_batch = self.get_frame(scene_index, state_index)

        else:
            # get the 100th frame (frame idx == 99) from the scene `index`
            data_batch = self.get_frame(index, 99)

        # add list of valid (for evaluation) agent IDs
        data_batch["all_valid_agents_track_ids"] = np.zeros(50)  # 0 means invalid
        if self.eval_mode:
            # take the only agent indices for the current scene
            valid_agent_indices = [self.agents_indices[agent_idx] for agent_idx in self.get_scene_indices(index)]
            # take the agent track ids corresponding to the valid agent indices
            valid_agent_track_ids = [self.dataset.agents[agent_idx]["track_id"] for agent_idx in valid_agent_indices]
            # add the valid agent track ids to the data_batch for evaluation
            data_batch["all_valid_agents_track_ids"][:len(valid_agent_track_ids)] = valid_agent_track_ids

        # Add ego to the agents to include it in training
        data_batch["all_other_agents_history_positions"] = np.concatenate(
            (data_batch["history_positions"][None], data_batch["all_other_agents_history_positions"]), axis=0)
        data_batch["all_other_agents_history_yaws"] = np.concatenate(
            (data_batch["history_yaws"][None], data_batch["all_other_agents_history_yaws"]), axis=0)
        data_batch["all_other_agents_history_extents"] = np.concatenate(
            (data_batch["history_extents"][None], data_batch["all_other_agents_history_extents"]), axis=0)
        data_batch["all_other_agents_history_availability"] = np.concatenate(
            (data_batch["history_availabilities"][None], data_batch["all_other_agents_history_availability"]), axis=0)

        data_batch["all_other_agents_future_positions"] = np.concatenate(
            (data_batch["target_positions"][None], data_batch["all_other_agents_future_positions"]), axis=0)
        data_batch["all_other_agents_future_yaws"] = np.concatenate(
            (data_batch["target_yaws"][None], data_batch["all_other_agents_future_yaws"]), axis=0)
        data_batch["all_other_agents_future_extents"] = np.concatenate(
            (data_batch["target_extents"][None], data_batch["all_other_agents_future_extents"]), axis=0)
        data_batch["all_other_agents_future_availability"] = np.concatenate(
            (data_batch["target_availabilities"][None], data_batch["all_other_agents_future_availability"]), axis=0)

        data_batch["all_other_agents_types"] = np.concatenate(
            (np.array([PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]]),
             data_batch["all_other_agents_types"]), axis=0)

        data_batch["all_other_agents_track_ids"] = np.concatenate(
            (data_batch["track_id"][None], data_batch["all_other_agents_track_ids"]), axis=0)

        data_batch["other_agents_polyline"] = np.concatenate(
            (data_batch["agent_trajectory_polyline"][None], data_batch["other_agents_polyline"]), axis=0)
        data_batch["other_agents_polyline_availability"] = np.concatenate(
            (data_batch["agent_polyline_availability"][None], data_batch["other_agents_polyline_availability"]), axis=0)

        # Change agent future availability according to minimum agent history
        # limit target availability to agents that have at least MIN_FRAME_HISTORY history frames (+ 1 frame as current)
        hist_avail_mask = data_batch["all_other_agents_history_availability"].sum(-1) > AGENT_MIN_FRAME_HISTORY
        future_avail_mask = data_batch["all_other_agents_future_availability"].sum(-1) > AGENT_MIN_FRAME_FUTURE
        data_batch["all_other_agents_future_availability"] *= (hist_avail_mask * future_avail_mask)[:, None]

        return data_batch
