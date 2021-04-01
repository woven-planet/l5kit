from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from l5kit.geometry import transform_points, transform_point
from l5kit.rasterization import Rasterizer
from l5kit.rasterization.box_rasterizer import draw_boxes, get_ego_as_agent
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from l5kit.data import AGENT_DTYPE
from torch.utils.data.dataloader import default_collate
from collections import defaultdict

from l5kit.simulation.dataset import SimulationDataset
from torch.utils.data.dataloader import default_collate


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


@torch.no_grad()
def unroll(
    cfg: Dict,
    model_ego: torch.nn.Module,
    model_agents: torch.nn.Module,
    dataset: SimulationDataset,
    start_frame_index: int = 0
    ) -> None:

    # TODO: enable limit unroll
    end_frame_index = len(dataset)

    for frame_index in tqdm(range(start_frame_index, end_frame_index)):
        next_frame_index = frame_index + 1

        # AGENTS
        agents_input = dataset.rasterise_agents_frame_batch(frame_index)
        agents_input_dict = default_collate(list(agents_input.values()))
        agents_output_dict = model_agents(agents_input_dict)

        # EGO
        ego_input = dataset.rasterise_frame_batch(frame_index)
        ego_input_dict = default_collate(ego_input)
        ego_output_dict = model_ego(ego_input_dict)

        if next_frame_index != end_frame_index:
            update_agents(dataset, next_frame_index, agents_input_dict, agents_output_dict)
            update_ego(dataset, next_frame_index, ego_input_dict, ego_output_dict)


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

    dataset = SimulationDataset(EgoDataset(cfg, zarr_dt, rast), [5, 6, 7, 8])

    unroll(cfg, MockModel(), MockModel(), dataset)
