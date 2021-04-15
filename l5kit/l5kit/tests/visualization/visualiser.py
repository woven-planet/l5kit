import numpy as np

from l5kit.data import AGENT_DTYPE, filter_agents_by_frames, FRAME_DTYPE
from l5kit.simulation.unroll import UnrollInputOutput
from l5kit.visualization.visualiser.zarr_utils import _get_frame_trajectories, _get_in_out_as_trajectories


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
