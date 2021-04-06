from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import torch
from tqdm.auto import tqdm
from l5kit.geometry import transform_points
from l5kit.data import AGENT_DTYPE

from l5kit.dataset import EgoDataset
from l5kit.simulation.dataset import SimulationDataset
from torch.utils.data.dataloader import default_collate


class SimulationConfig(NamedTuple):
    use_ego_gt: bool
    use_agents_gt: bool
    disable_new_agents: bool
    distance_th_far: float
    distance_th_close: float
    start_frame_index: int = 0
    num_simulation_steps: Optional[int] = None


class SimulationLoop:
    def __init__(self, sim_cfg: SimulationConfig, model_ego: torch.nn.Module, model_agents: torch.nn.Module):
        self.sim_cfg = sim_cfg
        self.model_ego = model_ego
        self.model_agents = model_agents

    def unroll(self, dataset: EgoDataset, scene_indices: List[int]) -> SimulationDataset:
        """
        Simulate the given dataset for the given scene indices
        :param dataset: the EgoDataset
        :param scene_indices: the scene indices we want to simulate
        :return: the simulated dataset
        TODO: this should probably return something else
        """
        sim_dataset = SimulationDataset(dataset, scene_indices, self.sim_cfg.start_frame_index,
                                        self.sim_cfg.disable_new_agents, self.sim_cfg.distance_th_far,
                                        self.sim_cfg.distance_th_close)

        if self.sim_cfg.num_simulation_steps is None:
            range_unroll = range(self.sim_cfg.start_frame_index, len(sim_dataset))
        else:
            end_frame_index = min(len(sim_dataset), self.sim_cfg.num_simulation_steps + self.sim_cfg.start_frame_index)
            range_unroll = range(self.sim_cfg.start_frame_index, end_frame_index)

        for frame_index in tqdm(range_unroll):
            next_frame_index = frame_index + 1

            # AGENTS
            agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)
            agents_input_dict = default_collate(list(agents_input.values()))
            agents_output_dict = self.model_agents(agents_input_dict)

            # EGO
            ego_input = sim_dataset.rasterise_frame_batch(frame_index)
            ego_input_dict = default_collate(ego_input)
            ego_output_dict = self.model_ego(ego_input_dict)

            if next_frame_index != range_unroll.stop:
                if not self.sim_cfg.use_agents_gt:
                    # TODO: suboptimal as we're still doing forward
                    self.update_agents(sim_dataset, next_frame_index, agents_input_dict, agents_output_dict)
                if not self.sim_cfg.use_ego_gt:
                    # TODO: suboptimal as we're still doing forward
                    self.update_ego(sim_dataset, next_frame_index, ego_input_dict, ego_output_dict)

        return sim_dataset

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
        pred_yaws = yaw + output_dict["yaws"][:, 0]

        next_agents = np.zeros(len(yaw), dtype=AGENT_DTYPE)
        next_agents["centroid"] = pred_trs
        next_agents["yaw"] = pred_yaws
        next_agents["track_id"] = input_dict["track_id"]
        next_agents["extent"] = input_dict["extent"]
        next_agents["label_probabilities"][:, 3] = 1

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
        pred_yaws = yaw + output_dict["yaws"][:, :1]

        dataset.set_ego_for_frame(frame_idx, 0, pred_trs, pred_yaws)


from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.configs import load_config_data
from l5kit.rasterization import build_rasterizer
from l5kit.dataset import EgoDataset


class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()

    def forward(self, x: Dict[str, torch.Tensor]):
        centroids = x["centroid"]
        bs = len(centroids)

        positions = torch.zeros(bs, 12, 2, device=centroids.device)
        positions[..., 0] = 1

        yaws = torch.zeros(bs, 12, device=centroids.device)

        return {"positions": positions, "yaws": yaws}


if __name__ == '__main__':
    zarr_dt = ChunkedDataset("/tmp/l5kit_data/scenes/sample.zarr").open()
    print(zarr_dt)

    cfg = load_config_data(
        "/Users/lucabergamini/Desktop/l5kit/examples/agent_motion_prediction/agent_motion_config.yaml")

    cfg["raster_params"]["ego_center"] = (0.25, 0.5)
    cfg["raster_params"]["pixel_size"] = (0.5, 0.5)
    cfg["raster_params"]["raster_size"] = (224, 224)

    cfg["model_params"]["history_num_frames"] = 0
    cfg["model_params"]["future_num_frames"] = 12
    cfg["raster_params"]["map_type"] = "py_semantic"

    rast = build_rasterizer(cfg, LocalDataManager("/tmp/l5kit_data"))

    sim_cfg = SimulationConfig(use_ego_gt=False,
                               use_agents_gt=False,
                               disable_new_agents=False,
                               distance_th_far=10,
                               distance_th_close=30,
                               start_frame_index=0)
    sim = SimulationLoop(sim_cfg, MockModel(), MockModel())
    sim.unroll(EgoDataset(cfg, zarr_dt, rast), [5, 6, 7, 8])
