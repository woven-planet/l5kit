from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from l5kit.data import ChunkedDataset, filter_agents_by_track_id, get_frames_slice_from_scenes, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.geometry import rotation33_as_yaw, yaw_as_rotation33
from l5kit.rasterization import build_rasterizer
from l5kit.simulation.unroll import ClosedLoopSimulator, SimulationConfig, SimulationDataset, TrajectoryStateIndices


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


def test_unroll_invalid_input(ego_cat_dataset: EgoDataset) -> None:
    # try to use None models with wrong config
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)

    with pytest.raises(ValueError):
        ClosedLoopSimulator(sim_cfg, ego_cat_dataset, torch.device("cpu"), MockModel(), None)

    with pytest.raises(ValueError):
        ClosedLoopSimulator(sim_cfg, ego_cat_dataset, torch.device("cpu"), None, MockModel())

    with pytest.raises(ValueError):
        ClosedLoopSimulator(sim_cfg, ego_cat_dataset, torch.device("cpu"), None, None)


def test_unroll_none_input(ego_cat_dataset: EgoDataset) -> None:
    sim_cfg = SimulationConfig(use_ego_gt=True, use_agents_gt=False, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)
    sim = ClosedLoopSimulator(sim_cfg, ego_cat_dataset, torch.device("cpu"), None, MockModel())
    assert isinstance(sim.model_ego, torch.nn.Sequential)
    assert isinstance(sim.model_agents, MockModel)

    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)
    sim = ClosedLoopSimulator(sim_cfg, ego_cat_dataset, torch.device("cpu"), MockModel(), None)
    assert isinstance(sim.model_ego, MockModel)
    assert isinstance(sim.model_agents, torch.nn.Sequential)

    sim_cfg = SimulationConfig(use_ego_gt=True, use_agents_gt=True, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)
    sim = ClosedLoopSimulator(sim_cfg, ego_cat_dataset, torch.device("cpu"), None, None)
    assert isinstance(sim.model_ego, torch.nn.Sequential)
    assert isinstance(sim.model_agents, torch.nn.Sequential)


def test_unroll(zarr_cat_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict) -> None:
    rasterizer = build_rasterizer(cfg, dmg)

    # change the first yaw of scene 1
    # this will break if some broadcasting happens
    zarr_cat_dataset.frames = np.asarray(zarr_cat_dataset.frames)
    slice_frames = get_frames_slice_from_scenes(zarr_cat_dataset.scenes[1])
    rot = zarr_cat_dataset.frames[slice_frames.start]["ego_rotation"].copy()
    zarr_cat_dataset.frames[slice_frames.start]["ego_rotation"] = yaw_as_rotation33(rotation33_as_yaw(rot + 0.75))

    scene_indices = list(range(len(zarr_cat_dataset.scenes)))
    ego_dataset = EgoDataset(cfg, zarr_cat_dataset, rasterizer)

    # control only agents at T0, control them forever
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000, num_simulation_steps=10)

    # ego will move by 1 each time
    ego_model = MockModel(advance_x=1.0)

    # agents will move by 0.5 each time
    agents_model = MockModel(advance_x=0.5)

    sim = ClosedLoopSimulator(sim_cfg, ego_dataset, torch.device("cpu"), ego_model, agents_model)
    sim_outputs = sim.unroll(scene_indices)

    # check ego movement
    for sim_output in sim_outputs:
        ego_tr = sim_output.simulated_ego["ego_translation"][: sim_cfg.num_simulation_steps, :2]
        ego_dist = np.linalg.norm(np.diff(ego_tr, axis=0), axis=-1)
        assert np.allclose(ego_dist, 1.0)

        ego_tr = sim_output.simulated_ego_states[: sim_cfg.num_simulation_steps,
                                                 TrajectoryStateIndices.X: TrajectoryStateIndices.Y + 1]
        ego_dist = np.linalg.norm(np.diff(ego_tr.numpy(), axis=0), axis=-1)
        assert np.allclose(ego_dist, 1.0, atol=1e-3)

        # all rotations should be the same as the first one as the MockModel outputs 0 for that
        rots_sim = sim_output.simulated_ego["ego_rotation"][: sim_cfg.num_simulation_steps]
        r_rep = sim_output.recorded_ego["ego_rotation"][0]
        for r_sim in rots_sim:
            assert np.allclose(rotation33_as_yaw(r_sim), rotation33_as_yaw(r_rep), atol=1e-2)

        # all rotations should be the same as the first one as the MockModel outputs 0 for that
        rots_sim = sim_output.simulated_ego_states[: sim_cfg.num_simulation_steps, TrajectoryStateIndices.THETA]
        r_rep = sim_output.recorded_ego_states[0, TrajectoryStateIndices.THETA]
        for r_sim in rots_sim:
            assert np.allclose(r_sim, r_rep, atol=1e-2)

    # check agents movements
    for sim_output in sim_outputs:
        # we need to know which agents were controlled during simulation
        # TODO: this is not ideal, we should keep track of them through the simulation
        sim_dataset = SimulationDataset.from_dataset_indices(ego_dataset, [sim_output.scene_id], sim_cfg)
        sim_dataset.rasterise_agents_frame_batch(0)  # this will fill agents_tracked

        agents_tracks = [el[1] for el in sim_dataset._agents_tracked]
        for track_id in agents_tracks:
            states = sim_output.simulated_agents
            agents = filter_agents_by_track_id(states, track_id)[: sim_cfg.num_simulation_steps]
            agent_dist = np.linalg.norm(np.diff(agents["centroid"], axis=0), axis=-1)
            assert np.allclose(agent_dist, 0.5)


@pytest.mark.parametrize("frame_range",
                         [(1, 10), (10, 5), (240, None), pytest.param((250, 10), marks=pytest.mark.xfail)])
def test_unroll_subset(zarr_cat_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict,
                       frame_range: Tuple[int, int]) -> None:
    rasterizer = build_rasterizer(cfg, dmg)

    scene_indices = list(range(len(zarr_cat_dataset.scenes)))
    ego_dataset = EgoDataset(cfg, zarr_cat_dataset, rasterizer)

    # control only agents at T0, control them forever
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                               distance_th_close=1000, distance_th_far=1000,
                               num_simulation_steps=frame_range[1], start_frame_index=frame_range[0])

    # ego will move by 1 each time
    ego_model = MockModel(advance_x=1.0)

    # agents will move by 0.5 each time
    agents_model = MockModel(advance_x=0.5)

    sim = ClosedLoopSimulator(sim_cfg, ego_dataset, torch.device("cpu"), ego_model, agents_model)
    sim_outputs = sim.unroll(scene_indices)

    for sim_out in sim_outputs:
        assert zarr_cat_dataset.frames[0] != sim_out.recorded_dataset.dataset.frames[0]
        assert zarr_cat_dataset.frames[0] != sim_out.simulated_dataset.dataset.frames[0]

        for ego_in_out in sim_out.ego_ins_outs:
            assert "positions" in ego_in_out.outputs and "yaws" in ego_in_out.outputs
            assert np.allclose(ego_in_out.outputs["positions"][:, 0], 1.0)
            assert np.allclose(ego_in_out.outputs["positions"][:, 1], 0.0)

        for agents_in_out in sim_out.agents_ins_outs:
            for agent_in_out in agents_in_out:
                assert "positions" in agent_in_out.outputs and "yaws" in agent_in_out.outputs
                assert np.allclose(agent_in_out.outputs["positions"][:, 0], 0.5)
                assert np.allclose(agent_in_out.outputs["positions"][:, 1], 0.0)

        if None not in frame_range:
            assert len(sim_out.recorded_dataset.dataset.frames) == frame_range[1]
            assert len(sim_out.simulated_dataset.dataset.frames) == frame_range[1]
            assert len(sim_out.simulated_ego_states) == frame_range[1]
            assert len(sim_out.recorded_ego_states) == frame_range[1]
            assert len(sim_out.recorded_ego) == frame_range[1]
            assert len(sim_out.simulated_ego) == frame_range[1]
            assert len(sim_out.ego_ins_outs) == len(sim_out.agents_ins_outs) == frame_range[1]

        ego_tr = sim_out.simulated_ego["ego_translation"][: sim_cfg.num_simulation_steps, :2]
        ego_dist = np.linalg.norm(np.diff(ego_tr, axis=0), axis=-1)
        assert np.allclose(ego_dist, 1.0)


def test_get_in_out_mock() -> None:
    with pytest.raises(ValueError):
        # repeated scene not allowed
        ClosedLoopSimulator.get_ego_in_out({"scene_index": np.ones(3)}, {"value": np.ones(3)})

    input_ego_dict = {"scene_index": np.asarray([0, 1, 2]), "track_id": np.full(3, -1), "mock": np.zeros(3)}
    output_ego_dict = {"output_value": input_ego_dict["scene_index"], "mock_out": np.zeros(3)}

    out = ClosedLoopSimulator.get_ego_in_out(input_ego_dict, output_ego_dict)

    # we should have 3 elements (one per scene)
    assert len(out) == 3
    for scene_index in input_ego_dict["scene_index"]:
        assert scene_index in out
        assert out[scene_index].track_id == -1
        assert out[scene_index].outputs["output_value"] == scene_index

    # we can test with exclude_keys also
    no_keys = {"mock", "mock_out"}
    out = ClosedLoopSimulator.get_ego_in_out(input_ego_dict, output_ego_dict, keys_to_exclude=no_keys)
    assert len(out) == 3
    for out_el in out.values():
        assert len(no_keys.intersection(out_el.inputs.keys())) == 0
        assert len(no_keys.intersection(out_el.outputs.keys())) == 0

    # for agents repeated scenes are allowed
    input_agents_dict = {"scene_index": np.asarray([0, 0, 2]), "track_id": np.asarray([0, 1, 0])}
    output_agents_dict = {"output_value_1": input_agents_dict["scene_index"],
                          "output_value_2": input_agents_dict["track_id"]}
    out = ClosedLoopSimulator.get_agents_in_out(input_agents_dict, output_agents_dict)

    assert len(out[0]) == 2
    assert len(out[2]) == 1

    for scene_index, scene_out in out.items():
        for agent_out in scene_out:
            assert agent_out.outputs["output_value_1"] == scene_index
            assert agent_out.outputs["output_value_2"] == agent_out.track_id
