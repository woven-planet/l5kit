from typing import Dict

import numpy as np
import torch

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from l5kit.environment import reward
from l5kit.environment.envs.l5_env import L5Env, SimulationConfigGym
from l5kit.geometry import rotation33_as_yaw
from l5kit.rasterization import build_rasterizer
from l5kit.simulation.unroll import TrajectoryStateIndices


class MockModel(torch.nn.Module):
    def __init__(self, advance_x: float = 0.0):
        super(MockModel, self).__init__()
        self.advance_x = advance_x

    def forward(self, x: Dict[str, np.ndarray]) -> np.ndarray:
        positions_and_yaws = torch.zeros(3,)
        positions_and_yaws[..., 0] = self.advance_x

        return positions_and_yaws.cpu().numpy()


def test_default_attributes(dmg: LocalDataManager, env_cfg_path: str) -> None:
    env = L5Env(env_cfg_path, dmg)
    assert isinstance(env.reward, reward.L2DisplacementYawReward)
    assert isinstance(env.sim_cfg, SimulationConfigGym)


def test_env_reset(dmg: LocalDataManager, env_cfg_path: str) -> None:
    env = L5Env(env_cfg_path, dmg)

    # env config
    cfg = load_config_data(env_cfg_path)
    # rasterisation
    rasterizer = build_rasterizer(cfg, dmg)
    raster_size = cfg["raster_params"]["raster_size"][0]
    n_channels = rasterizer.num_channels()

    # check first observation
    obs = env.reset()
    assert isinstance(obs, dict)
    assert len(obs) == 1
    assert 'image' in obs.keys()
    assert obs['image'].shape == (n_channels, raster_size, raster_size)


def test_env_episode(dmg: LocalDataManager, env_cfg_path: str) -> None:
    env = L5Env(env_cfg_path, dmg, rescale_action=False, return_info=True)

    # ego will move by 1 each time
    ego_model = MockModel(advance_x=1.0)

    # Unroll epsiode
    obs = env.reset()
    epsiode_len: int = 0
    for frame_idx in range(100):
        epsiode_len += 1
        action = ego_model(obs)
        obs, _, done, info = env.step(action)
        if done:
            sim_outputs = info["sim_outs"]
            break

    assert epsiode_len == (len(env.sim_dataset) - 2)

    # check ego movement
    for sim_output in sim_outputs:
        ego_tr = sim_output.simulated_ego["ego_translation"][1: env.sim_cfg.num_simulation_steps, :2]
        ego_dist = np.linalg.norm(np.diff(ego_tr, axis=0), axis=-1)
        assert np.allclose(ego_dist, 1.0)

        ego_tr = sim_output.simulated_ego_states[1: env.sim_cfg.num_simulation_steps,
                                                 TrajectoryStateIndices.X: TrajectoryStateIndices.Y + 1]
        ego_dist = np.linalg.norm(np.diff(ego_tr.numpy(), axis=0), axis=-1)
        assert np.allclose(ego_dist, 1.0, atol=1e-3)

        # all rotations should be the same as the first one as the MockModel outputs 0 for that
        rots_sim = sim_output.simulated_ego["ego_rotation"][1: env.sim_cfg.num_simulation_steps]
        r_rep = sim_output.recorded_ego["ego_rotation"][0]
        for r_sim in rots_sim:
            assert np.allclose(rotation33_as_yaw(r_sim), rotation33_as_yaw(r_rep), atol=1e-2)

        # all rotations should be the same as the first one as the MockModel outputs 0 for that
        rots_sim = sim_output.simulated_ego_states[1: env.sim_cfg.num_simulation_steps, TrajectoryStateIndices.THETA]
        r_rep = sim_output.recorded_ego_states[0, TrajectoryStateIndices.THETA]
        for r_sim in rots_sim:
            assert np.allclose(r_sim, r_rep, atol=1e-2)
