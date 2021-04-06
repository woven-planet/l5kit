from typing import Dict

import numpy as np
import torch

from l5kit.data import ChunkedDataset, filter_agents_by_track_id, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.simulation.unroll import SimulationConfig, SimulationLoop


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
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)

    # ego will move by 1 each time
    ego_model = MockModel(advance_x=1.0)

    # agents will move by 0.5 each time
    agents_model = MockModel(advance_x=0.5)

    sim = SimulationLoop(sim_cfg, ego_model, agents_model)
    sim_dataset = sim.unroll(ego_dataset, scene_indices)

    # check ego movement
    for dt_simulated in sim_dataset.scene_dataset_batch.values():
        ego_tr = dt_simulated.dataset.frames["ego_translation"][: sim_cfg.num_simulation_steps, :2]
        ego_dist = np.linalg.norm(np.diff(ego_tr, axis=0), axis=-1)
        assert np.allclose(ego_dist, 1.0)

    # check agents movements
    agents_tracks = [el[1] for el in sim_dataset.agents_tracked]
    for dt_simulated in sim_dataset.scene_dataset_batch.values():
        for track_id in agents_tracks:
            agents = filter_agents_by_track_id(dt_simulated.dataset.agents, track_id)[: sim_cfg.num_simulation_steps]
            agent_dist = np.linalg.norm(np.diff(agents["centroid"], axis=0), axis=-1)
            assert np.allclose(agent_dist, 0.5)
