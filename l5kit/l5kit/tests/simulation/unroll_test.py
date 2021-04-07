from typing import Dict

import numpy as np
import pytest
import torch

from l5kit.data import ChunkedDataset, filter_agents_by_track_id, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.simulation.unroll import SimulationConfig, SimulationLoop, SimulationDataset


@pytest.fixture(scope="function")
def ego_dataset(cfg: dict, dmg: LocalDataManager, zarr_cat_dataset: ChunkedDataset) -> EgoDataset:
    rasterizer = build_rasterizer(cfg, dmg)
    return EgoDataset(cfg, zarr_cat_dataset, rasterizer)


class MockModel(torch.nn.Module):
    def __init__(self, advance_x: float = 0.0):
        super(MockModel, self).__init__()
        self.advance_x = advance_x

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        centroids = x["centroid"]
        bs = len(centroids)

        positions = torch.zeros(bs, 12, 2, device=centroids.device)
        positions[..., 0] = self.advance_x

        yaws = torch.zeros(bs, 12, 1, device=centroids.device)

        return {"positions": positions, "yaws": yaws}


def test_unroll_invalid_input(ego_dataset: ego_dataset) -> None:
    # try to use None models with wrong config
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)

    with pytest.raises(ValueError):
        SimulationLoop(sim_cfg, ego_dataset, MockModel(), None)

    with pytest.raises(ValueError):
        SimulationLoop(sim_cfg, ego_dataset, None, MockModel())

    with pytest.raises(ValueError):
        SimulationLoop(sim_cfg, ego_dataset, None, None)


def test_unroll_none_input() -> None:
    sim_cfg = SimulationConfig(use_ego_gt=True, use_agents_gt=False, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)
    SimulationLoop(sim_cfg, ego_dataset, None, MockModel())

    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)
    SimulationLoop(sim_cfg, ego_dataset, MockModel(), None)

    sim_cfg = SimulationConfig(use_ego_gt=True, use_agents_gt=True, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)
    SimulationLoop(sim_cfg, ego_dataset, None, None)


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

    sim = SimulationLoop(sim_cfg, ego_dataset, ego_model, agents_model)
    sim_outputs = sim.unroll(scene_indices)

    # check ego movement
    for sim_output in sim_outputs:
        ego_tr = sim_output.simulated_ego_states["ego_translation"][: sim_cfg.num_simulation_steps, :2]
        ego_dist = np.linalg.norm(np.diff(ego_tr, axis=0), axis=-1)
        assert np.allclose(ego_dist, 1.0)

    # check agents movements
    for sim_output in sim_outputs:
        # we need to know which agents were controlled during simulation
        # TODO: this is not ideal, we should keep track of them through the simulation
        sim_dataset = SimulationDataset(ego_dataset, [sim_output.scene_id], sim_cfg.start_frame_index,
                                        sim_cfg.disable_new_agents, sim_cfg.distance_th_far,
                                        sim_cfg.distance_th_close)
        sim_dataset.rasterise_agents_frame_batch(0)  # this will fill agents_tracked

        agents_tracks = [el[1] for el in sim_dataset.agents_tracked]
        for track_id in agents_tracks:
            agents = filter_agents_by_track_id(sim_output.simulated_agents_states, track_id)[: sim_cfg.num_simulation_steps]
            agent_dist = np.linalg.norm(np.diff(agents["centroid"], axis=0), axis=-1)
            assert np.allclose(agent_dist, 0.5)
