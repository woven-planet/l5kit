from typing import Dict

import numpy as np
import torch

from l5kit.data import ChunkedDataset, filter_agents_by_track_id, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.simulation.dataset import SimulationDataset
from l5kit.simulation.unroll import unroll


class MockModel(torch.nn.Module):
    def __init__(self, advance_x: float = 0.0):
        super(MockModel, self).__init__()
        self.advance_x = advance_x

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        centroids = x["centroid"]
        bs = len(centroids)

        positions = torch.zeros(bs, 12, 2, device=centroids.device)
        positions[..., 0] = self.advance_x

        yaws = torch.zeros(bs, 12, device=centroids.device)

        return {"positions": positions, "yaws": yaws}


def test_unroll(zarr_cat_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict) -> None:
    rasterizer = build_rasterizer(cfg, dmg)

    scene_indices = list(range(len(zarr_cat_dataset.scenes)))
    ego_dataset = EgoDataset(cfg, zarr_cat_dataset, rasterizer)

    # control only agents at T0, control them forever
    dataset = SimulationDataset(ego_dataset, scene_indices, distance_th_close=1000,
                                distance_th_far=1000, disable_new_agents=True)

    # ego will move by 1 each time
    ego_model = MockModel(advance_x=1.0)

    # agents will move by 0.5 each time
    agents_model = MockModel(advance_x=0.5)

    simulation_steps = 10
    unroll(cfg, ego_model, agents_model, dataset, end_frame_index=simulation_steps)

    # TODO: change this when unroll has a return
    # check ego movement
    for dt_simulated in dataset.scene_dataset_batch.values():
        ego_tr = dt_simulated.dataset.frames["ego_translation"][: simulation_steps, :2]
        ego_dist = np.linalg.norm(np.diff(ego_tr, axis=0), axis=-1)
        assert np.allclose(ego_dist, 1.0)

    # check agents movements
    agents_tracks = [el[1] for el in dataset.agents_tracked]
    for dt_simulated in dataset.scene_dataset_batch.values():
        for track_id in agents_tracks:
            agents = filter_agents_by_track_id(dt_simulated.dataset.agents, track_id)[: simulation_steps]
            agent_dist = np.linalg.norm(np.diff(agents["centroid"], axis=0), axis=-1)
            assert np.allclose(agent_dist, 0.5)
