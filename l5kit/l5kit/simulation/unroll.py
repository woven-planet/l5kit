from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

from l5kit.data import AGENT_DTYPE, PERCEPTION_LABEL_TO_INDEX
from l5kit.dataset import EgoDataset
from l5kit.geometry import transform_points
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset


class SimulationOutput:
    def __init__(self, scene_id: int, sim_dataset: SimulationDataset):
        """This object holds information about the result of the simulation loop
        for a given scene dataset

        :param scene_id: the scene indices
        :param sim_dataset: the simulation dataset
        """
        if scene_id not in sim_dataset.scene_dataset_batch:
            raise ValueError(f"scene: {scene_id} not in {sim_dataset.scene_dataset_batch}")

        self.scene_id = scene_id
        self.recorded_dataset = sim_dataset.recorded_scene_dataset_batch[scene_id]
        self.simulated_dataset = sim_dataset.scene_dataset_batch[scene_id]

        self.recorded_ego_states = self.recorded_dataset.dataset.frames
        self.recorded_agents_states = self.recorded_dataset.dataset.agents
        self.simulated_ego_states = self.simulated_dataset.dataset.frames
        self.simulated_agents_states = self.simulated_dataset.dataset.agents

    def get_scene_id(self) -> int:
        """
        Get the scene index for this SimulationOutput

        :return: the scene index
        """
        return self.scene_id


class ClosedLoopSimulator:
    def __init__(self, sim_cfg: SimulationConfig, dataset: EgoDataset,
                 device: torch.device,
                 model_ego: Optional[torch.nn.Module] = None,
                 model_agents: Optional[torch.nn.Module] = None):
        """
        Create a simulation loop object capable of unrolling ego and agents
        :param sim_cfg: configuration for unroll
        :param dataset: EgoDataset used while unrolling
        :param device: a torch device. Inference will be performed here
        :param model_ego: the model to be used for ego
        :param model_agents: the model to be used for agents
        """
        self.sim_cfg = sim_cfg
        if not sim_cfg.use_ego_gt and model_ego is None:
            raise ValueError("ego model should not be None when simulating ego")
        if not sim_cfg.use_agents_gt and model_agents is None:
            raise ValueError("agents model should not be None when simulating agent")

        self.model_ego = torch.nn.Sequential().to(device) if model_ego is None else model_ego.to(device)
        self.model_agents = torch.nn.Sequential().to(device) if model_agents is None else model_agents.to(device)

        self.device = device
        self.dataset = dataset

    def unroll(self, scene_indices: List[int]) -> List[SimulationOutput]:
        """
        Simulate the dataset for the given scene indices
        :param scene_indices: the scene indices we want to simulate
        :return: the simulated dataset
        """
        sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_indices, self.sim_cfg)

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
                if len(agents_input):  # agents may not be available
                    agents_input_dict = default_collate(list(agents_input.values()))
                    agents_input_dict = {k: v.to(self.device) for k, v in agents_input_dict.items()}
                    agents_output_dict = self.model_agents(agents_input_dict)
                    if should_update:
                        self.update_agents(sim_dataset, next_frame_index, agents_input_dict, agents_output_dict)

            # EGO
            if not self.sim_cfg.use_ego_gt:
                ego_input = sim_dataset.rasterise_frame_batch(frame_index)
                ego_input_dict = default_collate(ego_input)
                ego_input_dict = {k: v.to(self.device) for k, v in ego_input_dict.items()}
                ego_output_dict = self.model_ego(ego_input_dict)
                if should_update:
                    self.update_ego(sim_dataset, next_frame_index, ego_input_dict, ego_output_dict)

        simulated_outputs: List[SimulationOutput] = []
        for scene_idx in scene_indices:
            simulated_outputs.append(SimulationOutput(scene_idx, sim_dataset))
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

        agents_update_dict: Dict[Tuple[int, int], np.ndarray] = {}
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
