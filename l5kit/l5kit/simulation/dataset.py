from copy import deepcopy
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np

from l5kit.data import filter_agents_by_frames, PERCEPTION_LABEL_TO_INDEX
from l5kit.dataset import EgoDataset
from l5kit.geometry.transform import yaw_as_rotation33
from l5kit.simulation.utils import disable_agents, insert_agent


class SimulationConfig(NamedTuple):
    """ Defines the parameters used for the simulation of ego and agents around it.

    :param use_ego_gt: whether to use GT annotations for ego instead of model's outputs
    :param use_agents_gt: whether to use GT annotations for agents instead of model's outputs
    :param disable_new_agents: whether to disable agents that are not returned at start_frame_index
    :param distance_th_far: if a tracked agent is closed than this value to ego, it will be controlled
    :param distance_th_close: if a new agent is closer than this value to ego, it will be controlled
    :param start_frame_index: the start index of the simulation
    :param num_simulation_steps: the number of step to simulate
    """
    use_ego_gt: bool
    use_agents_gt: bool
    disable_new_agents: bool
    distance_th_far: float
    distance_th_close: float
    start_frame_index: int = 0
    num_simulation_steps: Optional[int] = None


class SimulationDataset:
    def __init__(self, scene_dataset_batch: Dict[int, EgoDataset], sim_cfg: SimulationConfig) -> None:
        """This class allows to:
        - rasterise the same frame across multiple scenes for ego;
        - rasterise the same frame across multiple scenes for multiple agents;
        - filter agents based on distance to ego;
        - set ego in future frames;
        - set agents in future frames;

        .. note:: only vehicles (car label) are picked as agents

        :param scene_dataset_batch: a mapping from scene index to EgoDataset
        :param sim_cfg: the simulation config
        """
        if not len(scene_dataset_batch):
            raise ValueError("can't build a simulation dataset with an empty batch")
        self.scene_dataset_batch: Dict[int, EgoDataset] = scene_dataset_batch

        self._len = min([len(scene_dt.dataset.frames) for scene_dt in self.scene_dataset_batch.values()])

        self.sim_cfg = sim_cfg

        # buffer used to keep track of tracked agents during unroll as tuples of scene_idx, agent_idx
        self._agents_tracked: Set[Tuple[int, int]] = set()

        if self.sim_cfg.disable_new_agents:
            # we disable all agents that wouldn't be picked at start_frame
            for scene_idx, dt_ego in self.scene_dataset_batch.items():
                dataset_zarr = dt_ego.dataset
                frame = dataset_zarr.frames[self.sim_cfg.start_frame_index]
                ego_pos = frame["ego_translation"][:2]
                agents = dataset_zarr.agents
                frame_agents = filter_agents_by_frames(frame, agents)[0]
                frame_agents = self._filter_agents(scene_idx, frame_agents, ego_pos)
                disable_agents(dataset_zarr, allowlist=frame_agents["track_id"])

        # keep track of original dataset
        self.recorded_scene_dataset_batch = deepcopy(self.scene_dataset_batch)

    @staticmethod
    def from_dataset_indices(dataset: EgoDataset, scene_indices: List[int],
                             sim_cfg: SimulationConfig) -> "SimulationDataset":
        """Create a SimulationDataset by picking indices from the provided dataset

        :param dataset: the EgoDataset
        :param scene_indices: scenes from the EgoDataset to pick
        :param sim_cfg: a simulation config
        :return: the new SimulationDataset
        """
        if np.any(np.asarray(scene_indices) >= len(dataset.dataset.scenes)):
            raise ValueError(
                f"can't pick indices {scene_indices} from dataset with length: {len(dataset.dataset.scenes)}")

        scene_dataset_batch: Dict[int, EgoDataset] = {}  # dicts preserve insertion order
        for scene_idx in scene_indices:
            scene_dataset = dataset.get_scene_dataset(scene_idx)
            scene_dataset_batch[scene_idx] = scene_dataset
        return SimulationDataset(scene_dataset_batch, sim_cfg)

    def __len__(self) -> int:
        """
        Return the minimum number of frames across scenes

        :return: the number of frames
        """
        return self._len

    def rasterise_frame_batch(self, state_index: int) -> List[Dict[str, np.ndarray]]:
        """
        Get a frame from all scenes

        :param state_index: the frame index
        :return: a list of dict from EgoDatasets
        """
        frame_batch = [scene_dt[state_index] for scene_dt in self.scene_dataset_batch.values()]
        return frame_batch

    def set_ego(self, state_index: int, output_index: int, ego_translations: np.ndarray,
                ego_yaws: np.ndarray) -> None:
        """Mutate future frame position and yaw for ego across scenes. This acts on the underlying dataset

        :param state_index: the frame index to mutate
        :param output_index: the index in ego_translations and ego_yaws to use
        :param ego_translations: output translations (N, T, 2)
        :param ego_yaws: output yaws (N, T)
        """

        if len(ego_translations) != len(ego_yaws):
            raise ValueError("lengths mismatch between translations and yaws")
        if len(ego_translations) != len(self.scene_dataset_batch):
            raise ValueError("lengths mismatch between scenes and predictions")
        if state_index >= len(self):
            raise ValueError(f"trying to mutate frame:{state_index} but length is:{len(self)}")

        position_m_batch = ego_translations[:, output_index, :]
        angle_rad_batch = ego_yaws[:, output_index]
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
        for scene_index in self.scene_dataset_batch:
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

    def _update_agent_infos(self, scene_index: int, agent_track_ids: np.ndarray) -> None:
        """Update tracked agents object such that:
        - if agent was not there -> add it
        - if agent is not here anymore -> remove it
        This will be used next frame to control thresholds

        :param scene_index: index of the scene
        :param agent_track_ids: agents track ids for this frame
        """
        agent_track_set = set([(scene_index, int(track_id)) for track_id in agent_track_ids])

        self._agents_tracked.update(agent_track_set)

        remove_els = set([k for k in self._agents_tracked if k[0] == scene_index]) - agent_track_set
        for indices in remove_els:
            self._agents_tracked.remove(indices)

    def _filter_agents(self, scene_idx: int, frame_agents: np.ndarray,
                       ego_pos: np.ndarray) -> np.ndarray:
        """Filter agents according to a set of rules:
        if new agent (not in tracked_agents) then:
            - must be a car
            - must be in distance_th_close
        if tracked agent:
            - must be in distance_th_far

        This is to avoid acquiring and releasing the same agents if it is on the boundary of the selection

        :param scene_idx: the scene index (used to check for agents_infos)
        :param frame_agents: the agents in this frame
        :param ego_pos: the ego position in this frame
        :return: the filtered agents
        """
        # keep only vehicles
        car_index = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
        vehicle_mask = frame_agents["label_probabilities"][:, car_index]

        dt_agents_ths = self.scene_dataset_batch[scene_idx].cfg["raster_params"]["filter_agents_threshold"]
        vehicle_mask = vehicle_mask > dt_agents_ths
        frame_agents = frame_agents[vehicle_mask]

        distance_mask = np.zeros(len(frame_agents), dtype=np.bool)
        for idx_agent, agent in enumerate(frame_agents):
            track_id = int(agent["track_id"])

            distance = np.linalg.norm(ego_pos - agent["centroid"])
            # for distance use two thresholds
            if (scene_idx, track_id) in self._agents_tracked:
                # if we're already controlling this agent, th_far
                distance_mask[idx_agent] = distance < self.sim_cfg.distance_th_far
            else:
                # if not, start controlling it only if in th_close
                distance_mask[idx_agent] = distance < self.sim_cfg.distance_th_close

        return frame_agents[distance_mask]
