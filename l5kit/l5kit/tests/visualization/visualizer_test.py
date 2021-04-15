from collections import defaultdict

import numpy as np
import pytest

from l5kit.data import (AGENT_DTYPE, filter_agents_by_frames, FRAME_DTYPE, get_agents_slice_from_frames,
                        get_tl_faces_slice_from_frames)
from l5kit.dataset import EgoDataset
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import SimulationOutput, UnrollInputOutput
from l5kit.visualization.visualizer.common import FrameVisualization
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.visualization.visualizer.zarr_utils import (_get_frame_data, _get_frame_trajectories,
                                                       _get_in_out_as_trajectories, simulation_out_to_visualizer_scene)


def test_frame_trajectories_mock() -> None:
    # test a mock dataset
    ego_translation = np.asarray([(10, 10, 10), (20, 10, 20), (30, 30, 10)])
    agent_1_translation = np.asarray([(10, 10), (20, 20), (30, 30)])

    frames = np.zeros(3, dtype=FRAME_DTYPE)
    frames[0]["agent_index_interval"] = (0, 2)
    frames[1]["agent_index_interval"] = (2, 4)
    frames[2]["agent_index_interval"] = (4, 5)
    frames["ego_translation"] = ego_translation

    agents = np.zeros(5, dtype=AGENT_DTYPE)
    agents["track_id"] = [1, 2, 1, 2, 1]
    agents["centroid"][[0, 2, 4]] = agent_1_translation
    agents_frames = filter_agents_by_frames(frames, agents)

    trajs = _get_frame_trajectories(frames, agents_frames, np.asarray([1]), 0)

    assert len(trajs) == 2  # agent + ego
    assert np.allclose(trajs[0].xs, agent_1_translation[:, 0])
    assert np.allclose(trajs[0].ys, agent_1_translation[:, 1])
    assert np.allclose(trajs[0].track_id, 1)

    assert np.allclose(trajs[1].xs, ego_translation[:, 0])
    assert np.allclose(trajs[1].ys, ego_translation[:, 1])
    assert np.allclose(trajs[1].track_id, -1)


def test_get_inout_mock() -> None:
    inputs = {"target_positions": np.asarray([(0, 0), (1, 1), (2, 2)]), "world_from_agent": np.eye(3),
              "target_availabilities": np.asarray([1, 1, 0])}
    outputs = {"positions": np.asarray([(3, 3), (1, 1), (10, 10)])}

    unroll_inout = UnrollInputOutput(track_id=1, inputs=inputs, outputs=outputs)
    replay_traj, sim_traj = _get_in_out_as_trajectories(unroll_inout)

    assert len(replay_traj) == 2  # because of avail
    assert np.allclose(replay_traj, inputs["target_positions"][:len(replay_traj)])
    assert len(sim_traj) == 3  # not affected by avail
    assert np.allclose(sim_traj, outputs["positions"])


@pytest.mark.parametrize("frame_index", [0, 10, 100, 200])
def test_get_frame_data(ego_cat_dataset: EgoDataset, frame_index: int) -> None:
    mapAPI = ego_cat_dataset.rasterizer.sem_rast.mapAPI  # type: ignore

    frame = ego_cat_dataset.dataset.frames[frame_index]
    agent_slice = get_agents_slice_from_frames(frame)
    tls_slice = get_tl_faces_slice_from_frames(frame)
    agents = ego_cat_dataset.dataset.agents[agent_slice]
    tls = ego_cat_dataset.dataset.tl_faces[tls_slice]

    frame_out = _get_frame_data(mapAPI, frame, agents, tls)
    assert isinstance(frame_out, FrameVisualization)
    assert len(frame_out.agents) > 0
    assert len(frame_out.trajectories) == 0


def test_visualise(ego_cat_dataset: EgoDataset) -> None:
    mapAPI = ego_cat_dataset.rasterizer.sem_rast.mapAPI  # type: ignore
    sim_cfg = SimulationConfig(use_ego_gt=True, use_agents_gt=True, disable_new_agents=False,
                               distance_th_far=1, distance_th_close=0, num_simulation_steps=10)
    sim_dataset = SimulationDataset.from_dataset_indices(ego_cat_dataset, [0, 1], sim_cfg)
    sim_out = SimulationOutput(0, sim_dataset, ego_ins_outs=defaultdict(list),
                               agents_ins_outs=defaultdict(list))

    # ensure we can call the visualize
    visualize(0, simulation_out_to_visualizer_scene(sim_out, mapAPI))
