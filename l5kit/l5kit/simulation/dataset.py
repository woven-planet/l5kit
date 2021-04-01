from typing import Dict, List, Tuple, Set
from l5kit.geometry.transform import yaw_as_rotation33

import numpy as np

from l5kit.simulation.utils import disable_agents, insert_agent

from l5kit.data import filter_agents_by_frames, PERCEPTION_LABEL_TO_INDEX
from l5kit.dataset import EgoDataset
import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(self, dataset: EgoDataset, scene_indices: List[int], start_frame: int = 0,
                 disable_new_agents: bool = False, distance_th_far: float = 30,
                 distance_th_close: float = 10) -> None:
        """This class has all the functionalities of UnrollEgoDataset,
            but also grab and take care of agents around Ego

        :param dataset: the original ego dataset
        :param scene_indices: the indices of the scenes to take
        :param start_frame: the unroll start frame, use only if disable_new_agents is True
        :param disable_new_agents: if to disable new agents coming in from frame 1, defaults to False
        :param distance_th_far: the distance threshold for new agents,
            An agent will be grabbed if it's closer than this value to ego, defaults to 30
        :param distance_th_close: the distance threshold for already tracked agents,
            An agent will be grabbed if it's closer than this value to ego, defaults to 30
        """
        self.scene_indices = scene_indices
        self.filter_agents_thr = dataset.cfg["raster_params"]["filter_agents_threshold"]

        self.scene_dataset_batch: Dict[int, EgoDataset] = {}  # dicts preserve insertion order
        for scene_idx in self.scene_indices:
            scene_dataset = dataset.get_scene_dataset(scene_idx)
            self.scene_dataset_batch[scene_idx] = scene_dataset

        # agents stuff
        self.agents_tracked: Set[Tuple[int, int]] = set()

        self.distance_th_far = distance_th_far
        self.distance_th_close = distance_th_close
        self.disable_new_agents = disable_new_agents

        if disable_new_agents:
            for scene_index in scene_indices:
                dataset_zarr = self.scene_dataset_batch[scene_index].dataset
                frame = dataset_zarr.frames[start_frame]
                ego_pos = frame["ego_translation"][:2]
                agents = dataset_zarr.agents
                frame_agents = filter_agents_by_frames(frame, agents)[0]
                frame_agents = self._filter_agents(0, frame_agents, ego_pos)
                disable_agents(dataset_zarr, allowlist=frame_agents["track_id"])

    def __len__(self) -> int:
        """
        Return the minimum number of frames across scenes

        :return: the number of frames
        """
        num_frame_batch = [len(scene_dt.dataset.frames) for scene_dt in self.scene_dataset_batch.values()]
        return min(num_frame_batch)

    def __getitem__(self, indices: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Get a single frame from a single scene

        :param indices: tuple [scene_idx, frame_idx]
        :return: a dict from EgoDataset
        """
        scene_index, frame_index = indices
        data_batch = self.scene_dataset_batch[scene_index][frame_index]
        return data_batch

    def rasterise_frame_batch(self, state_index: int) -> List[np.ndarray]:
        """
        Get a frame from all scenes

        :param state_index: the frame index
        :return: a list of dict from EgoDatasets
        """
        frame_batch = [scene_dt.dataset.frames[state_index] for scene_dt in self.scene_dataset_batch.values()]
        return frame_batch

    def set_ego_for_frame(self, state_index: int, output_index: int, ego_translations: torch.Tensor,
                          ego_yaws: torch.Tensor) -> None:
        """Mutate future frame position and yaw for ego across scenes. This acts on the underlying dataset

        :param state_index: the frame index to mutate
        :param output_index: the index in ego_translations and ego_yaws to use
        :param ego_translations: output translations
        :param ego_yaws: output yaws
        :return:
        """

        """
        Mutate future frame position and yaw. This acts on the underlying dataset
        """
        if len(ego_translations) != len(ego_yaws):
            raise ValueError("lengths mismatch between translations and yaws")
        if len(ego_translations) != len(self.scene_indices):
            raise ValueError("lengths mismatch between scenes and predictions")
        if state_index >= len(self):
            raise ValueError(f"trying to mutate frame:{state_index} but length is:{len(self)}")

        position_m_batch = ego_translations[:, output_index, :].detach().cpu().numpy()
        angle_rad_batch = ego_yaws[:, output_index].detach().cpu().numpy()
        for i, (scene_dataset, position_m, angle_rad) in enumerate(
            zip(self.scene_dataset_batch.values(), position_m_batch, angle_rad_batch)
        ):
            scene_dataset.dataset.frames[state_index]["ego_translation"][:2] = position_m
            scene_dataset.dataset.frames[state_index]["ego_rotation"] = yaw_as_rotation33(angle_rad)

    def set_agents(self, state_index: int, agents_infos: Dict[Tuple[int, int], np.ndarray]) -> None:
        """Set multiple agents in the scene datasets.

        :param state_index: the frame index to set (same for all datasets)
        :param agents_infos: a dict mapping (scene_idx, agent_idx) to the agent array
        """
        for (scene_idx, _), agent in agents_infos.items():
            insert_agent(agent, state_index, self.scene_dataset_batch[scene_idx].dataset)

    def rasterise_agents_frame_batch(self, state_index: int) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
        """Rasterise agents for each scene in the batch at a given frame.

        :param state_index: the frame index in the scene
        :return: a dict mapping from [scene_id, track_id] to the numpy dict
        """
        ret = {}
        for scene_index in self.scene_indices:
            ret.update(self._rasterise_agents_frame(scene_index, state_index))
        return ret

    def _rasterise_agents_frame(self, scene_index: int,
                                state_index: int) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
        """Rasterise agents of interest for a given frame in a given scene.

        :param scene_index: index of the scene
        :param state_index: frame index
        :return: a dict mapping [scene_idx, agent_idx] to dict
        """
        # filter agents around ego based on distance and threshold
        dataset = self.scene_dataset_batch[scene_index]
        frame = dataset.dataset.frames[state_index]

        frame_agents = filter_agents_by_frames(frame, dataset.dataset.agents)[0]
        frame_agents = self._filter_agents(scene_index, frame_agents, frame["ego_translation"][:2])

        # rasterise individual agents
        agents_dict: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        for agent in frame_agents:
            track_id = int(agent["track_id"])
            el = dataset.get_frame(scene_index=0, state_index=state_index, track_id=track_id)
            # we replace the scene_index here to match the real one (otherwise is 0)
            el["scene_index"] = scene_index
            agents_dict[scene_index, track_id] = el

        self._update_agent_infos(scene_index, frame_agents["track_id"])
        return agents_dict

    def _update_agent_infos(self, scene_index: int, agent_track_ids: np.ndarray):
        """Update tracked agents object such that:
        - if agent was not there -> add it
        - if agent is not here anymore -> remove it
        This will be used next frame to control thresholds

        :param scene_index: index of the scene
        :param agent_track_ids: agents track ids for this frame
        """
        agent_track_set = set([(scene_index, int(track_id)) for track_id in agent_track_ids])

        self.agents_tracked.update(agent_track_set)

        remove_els = set([k for k in self.agents_tracked if k[0] == scene_index]) - agent_track_set
        for indices in remove_els:
            self.agents_tracked.remove(indices)

    def _filter_agents(self, scene_idx: int, frame_agents: np.ndarray,
                       ego_pos: np.ndarray) -> np.ndarray:
        """Filter agents according to a set of rules:
        if new agent (not in agents_infos) then:
            - must be a car
            - must be in distance_th_close
        if tracked agent:
            - must be in distance_th_far

        :param scene_idx: the scene index (used to check for agents_infos)
        :param frame_agents: the agents in this frame
        :param ego_pos: the ego position in this frame
        :return: the filtered agents
        """
        # keep only vehicles
        vehicle_mask = frame_agents["label_probabilities"][:,
                                     PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]] > self.filter_agents_thr
        frame_agents = frame_agents[vehicle_mask]

        # for distance use two thresholds
        distance_mask = np.zeros(len(frame_agents), dtype=np.bool)
        for idx_agent, agent in enumerate(frame_agents):
            track_id = int(agent["track_id"])

            distance = np.linalg.norm(ego_pos - agent["centroid"])
            if (scene_idx, track_id) in self.agents_tracked:
                # if we're already controlling this agent, th_far
                distance_mask[idx_agent] = distance < self.distance_th_far
            else:
                # if not, start controlling it only if in th_close
                distance_mask[idx_agent] = distance < self.distance_th_close

        return frame_agents[distance_mask]
