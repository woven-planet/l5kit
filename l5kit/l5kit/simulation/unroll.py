from collections import defaultdict
from enum import IntEnum
from typing import DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

from l5kit.data import AGENT_DTYPE, PERCEPTION_LABEL_TO_INDEX
from l5kit.dataset import EgoDataset
from l5kit.dataset.utils import move_to_device, move_to_numpy
from l5kit.geometry import rotation33_as_yaw, transform_points
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset


class TrajectoryStateIndices(IntEnum):
    """Defines indices for accessing trajectory states from LocalSceneBatch.
    example: all_speeds = local_scene_batch.recorded_ego_states[:, TrajectoryStateIndices.SPEED]

    :param X: the index for x position
    :param Y: the index for y position
    :param THETA: the index for the angle in radians
    :param SPEED: the index for speed in mps
    :param ACCELERATION: the index for acceleration in mps^2
    :param CURVATURE: the index for curvature (1pm)
    :param TIME: the index for time (seconds)
    """
    X = 0
    Y = 1
    THETA = 2
    SPEED = 3
    ACCELERATION = 4
    CURVATURE = 5
    TIME = 6


class UnrollInputOutput(NamedTuple):
    """ A single input output dict for an agent/ego in a given frame

    :param track_id: the agent track id
    :param inputs: input dict
    :param outputs: output dict
    """
    track_id: int
    inputs: Dict[str, np.ndarray]
    outputs: Dict[str, np.ndarray]


class ClosedLoopSimulatorModes(IntEnum):
    """Defines the different modes for which the closed loop simulator can be used.

    :param L5KIT: the index for using closed loop simulator for L5Kit environment
    :param GYM: the index for using closed loop simulator for Gym environment
    """
    L5KIT = 0
    GYM = 1


class SimulationOutputCLE:
    def __init__(self, scene_id: int, sim_dataset: SimulationDataset,
                 ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]],
                 agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]]):
        """This object holds information about the result of the simulation loop
        for a given scene dataset for close loop evaluation.

        :param scene_id: the scene indices
        :param sim_dataset: the simulation dataset
        :param ego_ins_outs: all inputs and outputs for ego (each frame of each scene has only one)
        :param agents_ins_outs: all inputs and outputs for agents (multiple per frame in a scene)
        """
        if scene_id not in sim_dataset.scene_dataset_batch:
            raise ValueError(f"scene: {scene_id} not in sim datasets: {sim_dataset.scene_dataset_batch}")

        self.scene_id = scene_id
        recorded_dataset = sim_dataset.recorded_scene_dataset_batch[scene_id]
        simulated_dataset = sim_dataset.scene_dataset_batch[scene_id]

        self.simulated_agents = simulated_dataset.dataset.agents
        self.recorded_agents = recorded_dataset.dataset.agents
        self.recorded_ego = recorded_dataset.dataset.frames
        self.simulated_ego = simulated_dataset.dataset.frames

        self.simulated_ego_states = self.build_trajectory_states(self.simulated_ego)
        self.recorded_ego_states = self.build_trajectory_states(self.recorded_ego)

        self.ego_ins_outs = ego_ins_outs[scene_id]
        self.agents_ins_outs = agents_ins_outs[scene_id]

    def get_scene_id(self) -> int:
        """
        Get the scene index for this SimulationOutput

        :return: the scene index
        """
        return self.scene_id

    @staticmethod
    def build_trajectory_states(frames: np.ndarray) -> torch.Tensor:
        """
        Convert frames into a torch trajectory
        :param frames: the scene frames
        :return: the trajectory
        """
        trajectory_states = torch.zeros(len(frames), len(TrajectoryStateIndices), dtype=torch.float)
        translations = frames["ego_translation"]
        rotations = frames["ego_rotation"]

        for idx_frame in range(len(frames)):
            # TODO: there is a conversion from float64 to float32 here
            trajectory_states[idx_frame, TrajectoryStateIndices.X] = translations[idx_frame, 0]
            trajectory_states[idx_frame, TrajectoryStateIndices.Y] = translations[idx_frame, 1]
            trajectory_states[idx_frame, TrajectoryStateIndices.THETA] = rotation33_as_yaw(rotations[idx_frame])
            # TODO: Replace 0.1 by step_time
            trajectory_states[idx_frame, TrajectoryStateIndices.TIME] = 0.1 * idx_frame
            # TODO: we may need to fill other fields

        return trajectory_states


class SimulationOutput(SimulationOutputCLE):
    def __init__(self, scene_id: int, sim_dataset: SimulationDataset,
                 ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]],
                 agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]]):
        """This object holds information about the result of the simulation loop
        for a given scene dataset for close loop evaluation and visualization

        :param scene_id: the scene indices
        :param sim_dataset: the simulation dataset
        :param ego_ins_outs: all inputs and outputs for ego (each frame of each scene has only one)
        :param agents_ins_outs: all inputs and outputs for agents (multiple per frame in a scene)
        """

        super(SimulationOutput, self).__init__(scene_id, sim_dataset, ego_ins_outs, agents_ins_outs)

        # Useful for visualization
        self.recorded_dataset = sim_dataset.recorded_scene_dataset_batch[scene_id]
        self.simulated_dataset = sim_dataset.scene_dataset_batch[scene_id]


class ClosedLoopSimulator:
    def __init__(self, sim_cfg: SimulationConfig, dataset: EgoDataset,
                 device: torch.device,
                 model_ego: Optional[torch.nn.Module] = None,
                 model_agents: Optional[torch.nn.Module] = None,
                 keys_to_exclude: Tuple[str] = ("image",),
                 mode: int = ClosedLoopSimulatorModes.L5KIT):
        """
        Create a simulation loop object capable of unrolling ego and agents
        :param sim_cfg: configuration for unroll
        :param dataset: EgoDataset used while unrolling
        :param device: a torch device. Inference will be performed here
        :param model_ego: the model to be used for ego
        :param model_agents: the model to be used for agents
        :param keys_to_exclude: keys to exclude from input/output (e.g. huge blobs)
        :param mode: the framework that uses the closed loop simulator
        """
        self.sim_cfg = sim_cfg
        if not sim_cfg.use_ego_gt and model_ego is None and mode == ClosedLoopSimulatorModes.L5KIT:
            raise ValueError("ego model should not be None when simulating ego")
        if not sim_cfg.use_agents_gt and model_agents is None and mode == ClosedLoopSimulatorModes.L5KIT:
            raise ValueError("agents model should not be None when simulating agent")
        if sim_cfg.use_ego_gt and mode == ClosedLoopSimulatorModes.GYM:
            raise ValueError("ego has to be simulated when using gym environment")
        if not sim_cfg.use_agents_gt and mode == ClosedLoopSimulatorModes.GYM:
            raise ValueError("agents need be log-replayed when using gym environment")

        self.model_ego = torch.nn.Sequential().to(device) if model_ego is None else model_ego.to(device)
        self.model_agents = torch.nn.Sequential().to(device) if model_agents is None else model_agents.to(device)

        self.device = device
        self.dataset = dataset

        self.keys_to_exclude = set(keys_to_exclude)

    def unroll(self, scene_indices: List[int]) -> List[SimulationOutput]:
        """
        Simulate the dataset for the given scene indices
        :param scene_indices: the scene indices we want to simulate
        :return: the simulated dataset
        """
        sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_indices, self.sim_cfg)

        agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        for frame_index in tqdm(range(len(sim_dataset)), disable=not self.sim_cfg.show_info):
            next_frame_index = frame_index + 1
            should_update = next_frame_index != len(sim_dataset)

            # AGENTS
            if not self.sim_cfg.use_agents_gt:
                agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)
                if len(agents_input):  # agents may not be available
                    agents_input_dict = default_collate(list(agents_input.values()))
                    agents_output_dict = self.model_agents(move_to_device(agents_input_dict, self.device))

                    # for update we need everything as numpy
                    agents_input_dict = move_to_numpy(agents_input_dict)
                    agents_output_dict = move_to_numpy(agents_output_dict)

                    if should_update:
                        self.update_agents(sim_dataset, next_frame_index, agents_input_dict, agents_output_dict)

                    # update input and output buffers
                    agents_frame_in_out = self.get_agents_in_out(agents_input_dict, agents_output_dict,
                                                                 self.keys_to_exclude)
                    for scene_idx in scene_indices:
                        agents_ins_outs[scene_idx].append(agents_frame_in_out.get(scene_idx, []))

            # EGO
            if not self.sim_cfg.use_ego_gt:
                ego_input = sim_dataset.rasterise_frame_batch(frame_index)
                ego_input_dict = default_collate(ego_input)
                ego_output_dict = self.model_ego(move_to_device(ego_input_dict, self.device))

                ego_input_dict = move_to_numpy(ego_input_dict)
                ego_output_dict = move_to_numpy(ego_output_dict)

                if should_update:
                    self.update_ego(sim_dataset, next_frame_index, ego_input_dict, ego_output_dict)

                ego_frame_in_out = self.get_ego_in_out(ego_input_dict, ego_output_dict, self.keys_to_exclude)
                for scene_idx in scene_indices:
                    ego_ins_outs[scene_idx].append(ego_frame_in_out[scene_idx])

        simulated_outputs: List[SimulationOutput] = []
        for scene_idx in scene_indices:
            simulated_outputs.append(SimulationOutput(scene_idx, sim_dataset, ego_ins_outs, agents_ins_outs))
        return simulated_outputs

    @staticmethod
    def update_agents(dataset: SimulationDataset, frame_idx: int, input_dict: Dict[str, np.ndarray],
                      output_dict: Dict[str, np.ndarray]) -> None:
        """Update the agents in frame_idx (across scenes) using agents_output_dict

        :param dataset: the simulation dataset
        :param frame_idx: index of the frame to modify
        :param input_dict: the input to the agent model
        :param output_dict: the output of the agent model
        :return:
        """

        agents_update_dict: Dict[Tuple[int, int], np.ndarray] = {}

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
    def get_agents_in_out(input_dict: Dict[str, np.ndarray],
                          output_dict: Dict[str, np.ndarray],
                          keys_to_exclude: Optional[Set[str]] = None) -> Dict[int, List[UnrollInputOutput]]:
        """Get all agents inputs and outputs as a dict mapping scene index to a list of UnrollInputOutput

        :param input_dict: all agent model inputs (across scenes)
        :param output_dict: all agent model outputs (across scenes)
        :param keys_to_exclude: if to drop keys from input/output (e.g. huge blobs)
        :return: the dict mapping scene index to a list UnrollInputOutput. Some scenes may be missing
        """
        key_required = {"track_id", "scene_index"}
        if len(key_required.intersection(input_dict.keys())) != len(key_required):
            raise ValueError(f"track_id and scene_index not found in keys {input_dict.keys()}")

        keys_to_exclude = keys_to_exclude if keys_to_exclude is not None else set()
        if len(key_required.intersection(keys_to_exclude)) != 0:
            raise ValueError(f"can't drop required keys: {keys_to_exclude}")

        ret_dict = defaultdict(list)
        for idx_agent in range(len(input_dict["track_id"])):
            agent_in = {k: v[idx_agent] for k, v in input_dict.items() if k not in keys_to_exclude}
            agent_out = {k: v[idx_agent] for k, v in output_dict.items() if k not in keys_to_exclude}
            ret_dict[agent_in["scene_index"]].append(UnrollInputOutput(track_id=agent_in["track_id"],
                                                                       inputs=agent_in,
                                                                       outputs=agent_out))
        return dict(ret_dict)

    @staticmethod
    def get_ego_in_out(input_dict: Dict[str, np.ndarray],
                       output_dict: Dict[str, np.ndarray],
                       keys_to_exclude: Optional[Set[str]] = None) -> Dict[int, UnrollInputOutput]:
        """Get all ego inputs and outputs as a dict mapping scene index to a single UnrollInputOutput

        :param input_dict: all ego model inputs (across scenes)
        :param output_dict: all ego model outputs (across scenes)
        :param keys_to_exclude: if to drop keys from input/output (e.g. huge blobs)
        :return: the dict mapping scene index to a single UnrollInputOutput.
        """
        key_required = {"track_id", "scene_index"}
        if len(key_required.intersection(input_dict.keys())) != len(key_required):
            raise ValueError(f"track_id and scene_index not found in keys {input_dict.keys()}")

        keys_to_exclude = keys_to_exclude if keys_to_exclude is not None else set()
        if len(key_required.intersection(keys_to_exclude)) != 0:
            raise ValueError(f"can't drop required keys: {keys_to_exclude}")

        ret_dict = {}
        scene_indices = input_dict["scene_index"]
        if len(np.unique(scene_indices)) != len(scene_indices):
            raise ValueError(f"repeated scene_index for ego! {scene_indices}")

        for idx_ego in range(len(scene_indices)):
            ego_in = {k: v[idx_ego] for k, v in input_dict.items() if k not in keys_to_exclude}
            ego_out = {k: v[idx_ego] for k, v in output_dict.items() if k not in keys_to_exclude}
            ret_dict[ego_in["scene_index"]] = UnrollInputOutput(track_id=ego_in["track_id"],
                                                                inputs=ego_in,
                                                                outputs=ego_out)
        return ret_dict

    @staticmethod
    def update_ego(dataset: SimulationDataset, frame_idx: int, input_dict: Dict[str, np.ndarray],
                   output_dict: Dict[str, np.ndarray]) -> None:
        """Update ego across scenes for the given frame index.

        :param dataset: The simulation dataset
        :param frame_idx: index of the frame to modify
        :param input_dict: the input to the ego model
        :param output_dict: the output of the ego model
        :return:
        """
        world_from_agent = input_dict["world_from_agent"]
        yaw = input_dict["yaw"]
        pred_trs = transform_points(output_dict["positions"][:, :1], world_from_agent)
        pred_yaws = np.expand_dims(yaw, -1) + output_dict["yaws"][:, :1, 0]

        dataset.set_ego(frame_idx, 0, pred_trs, pred_yaws)
