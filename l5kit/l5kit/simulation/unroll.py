from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

from l5kit.data import AGENT_DTYPE, PERCEPTION_LABEL_TO_INDEX
from l5kit.dataset import EgoDataset
from l5kit.geometry import transform_points
from l5kit.simulation.dataset import SimulationDataset


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


class SimulationOutputs:
    def __init__(self, scene_id: int, sim_dataset: SimulationDataset):
        """This object holds information about the result of the simulation loop
        for a given scene dataset

        :param scene_id: the scene indices
        :param sim_dataset: the simulation dataset
        """
        if scene_id not in sim_dataset.scene_indices:
            raise ValueError(f"scene: {scene_id} not in {sim_dataset.scene_indices}")

        self.scene_id = scene_id
        self.recorded_dataset = sim_dataset.recorded_scene_dataset_batch[scene_id]
        self.simulated_dataset = sim_dataset.scene_dataset_batch[scene_id]

        self.recorded_ego_states = self.recorded_dataset.dataset.frames
        self.recorded_agents_states = self.recorded_dataset.dataset.agents
        self.simulated_ego_states = self.simulated_dataset.dataset.frames
        self.simulated_agents_states = self.simulated_dataset.dataset.agents

    def get_scene_id(self) -> int:
        """
        Get the scene index for this SimulationOutputs

        :return: the scene index
        """
        return self.scene_id


class SimulationLoop:
    def __init__(self, sim_cfg: SimulationConfig, dataset: EgoDataset,
                 model_ego: Optional[torch.nn.Module] = None,
                 model_agents: Optional[torch.nn.Module] = None):
        """
        Create a simulation loop object capable of unrolling ego and agents
        :param sim_cfg: configuration for unroll
        :param dataset: EgoDataset used while unrolling
        :param model_ego: the model to be used for ego
        :param model_agents: the model to be used for agents
        """
        self.sim_cfg = sim_cfg
        if not sim_cfg.use_ego_gt and model_ego is None:
            raise ValueError("ego model should not be None when simulating ego")
        if not sim_cfg.use_agents_gt and model_agents is None:
            raise ValueError("agents model should not be None when simulating agent")

        self.model_ego = model_ego
        self.model_agents = model_agents
        self.dataset = dataset

    def unroll(self, scene_indices: List[int]) -> List[SimulationOutputs]:
        """
        Simulate the dataset for the given scene indices
        :param scene_indices: the scene indices we want to simulate
        :return: the simulated dataset
        """
        sim_dataset = SimulationDataset(self.dataset, scene_indices, self.sim_cfg.start_frame_index,
                                        self.sim_cfg.disable_new_agents, self.sim_cfg.distance_th_far,
                                        self.sim_cfg.distance_th_close)

        if self.sim_cfg.num_simulation_steps is None:
            range_unroll = range(self.sim_cfg.start_frame_index, len(sim_dataset))
        else:
            end_frame_index = min(len(sim_dataset), self.sim_cfg.num_simulation_steps + self.sim_cfg.start_frame_index)
            range_unroll = range(self.sim_cfg.start_frame_index, end_frame_index)

        for frame_index in tqdm(range_unroll):
            next_frame_index = frame_index + 1
            should_update = next_frame_index != range_unroll.stop

            # AGENTS
            if not self.sim_cfg.use_agents_gt:
                agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)
                agents_input_dict = default_collate(list(agents_input.values()))
                agents_output_dict = self.model_agents(agents_input_dict)
                if should_update:
                    self.update_agents(sim_dataset, next_frame_index, agents_input_dict, agents_output_dict)

            # EGO
            if not self.sim_cfg.use_ego_gt:
                ego_input = sim_dataset.rasterise_frame_batch(frame_index)
                ego_input_dict = default_collate(ego_input)
                ego_output_dict = self.model_ego(ego_input_dict)
                if should_update:
                    self.update_ego(sim_dataset, next_frame_index, ego_input_dict, ego_output_dict)

        simulated_outputs: List[SimulationOutputs] = []
        for scene_idx in scene_indices:
            simulated_outputs.append(SimulationOutputs(scene_idx, sim_dataset))
        return simulated_outputs

    @staticmethod
    def update_agents(dataset: SimulationDataset, frame_idx: int, input_dict: Dict[str, torch.Tensor],
                      output_dict: Dict[str, torch.Tensor]) -> None:
        """Update the agents in frame_idx (across scenes) using agents_output_dict

        :param dataset: the simulation dataset
        :param frame_idx: index of the frame to modify
        :param input_dict: the input to the agent model
        :param output_dict: the output of the agent model
        :return:
        """

        agents_update_dict: Dict[Tuple[int, int]: np.ndarray] = {}
        input_dict = {k: v.cpu().numpy() for k, v in input_dict.items()}
        output_dict = {k: v.cpu().numpy() for k, v in output_dict.items()}

        world_from_agent = input_dict["world_from_agent"]
        yaw = input_dict["yaw"]
        pred_trs = transform_points(output_dict["positions"][:, :1], world_from_agent)[:, 0]
        pred_yaws = yaw + output_dict["yaws"][:, 0, 0]

        next_agents = np.zeros(len(yaw), dtype=AGENT_DTYPE)
        next_agents["centroid"] = pred_trs
        next_agents["yaw"] = pred_yaws
        next_agents["track_id"] = input_dict["track_id"]
        next_agents["extent"] = input_dict["extent"]

        next_agents["label_probabilities"][:, PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]] = 1

        for scene_idx, next_agent in zip(input_dict["scene_index"], next_agents):
            agents_update_dict[(scene_idx, next_agent["track_id"])] = np.expand_dims(next_agent, 0)
        dataset.set_agents(frame_idx, agents_update_dict)

    @staticmethod
    def update_ego(dataset: SimulationDataset, frame_idx: int, input_dict: Dict[str, torch.Tensor],
                   output_dict: Dict[str, torch.Tensor]) -> None:
        """Update ego across scenes for the given frame index.

        :param dataset: The simulation dataset
        :param frame_idx: index of the frame to modify
        :param input_dict: the input to the ego model
        :param output_dict: the output of the ego model
        :return:
        """
        input_dict = {k: v.cpu().numpy() for k, v in input_dict.items()}
        output_dict = {k: v.cpu().numpy() for k, v in output_dict.items()}

        world_from_agent = input_dict["world_from_agent"]
        yaw = input_dict["yaw"]
        pred_trs = transform_points(output_dict["positions"][:, :1], world_from_agent)
        pred_yaws = np.expand_dims(yaw, -1) + output_dict["yaws"][:, :1, 0]

        dataset.set_ego(frame_idx, 0, pred_trs, pred_yaws)
