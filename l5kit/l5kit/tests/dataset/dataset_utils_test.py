import numpy as np
import pytest

from l5kit.data import ChunkedDataset, get_agents_slice_from_frames, get_frames_slice_from_scenes, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer


@pytest.mark.parametrize("scene_idx", [0, pytest.param(999, marks=pytest.mark.xfail)])
def test_get_scene_indices_ego(scene_idx: int, zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"
    rasterizer = build_rasterizer(cfg, dmg)
    dataset = EgoDataset(cfg, zarr_dataset, rasterizer)

    scene_indices = dataset.get_scene_indices(scene_idx)
    frame_slice = get_frames_slice_from_scenes(zarr_dataset.scenes[scene_idx])
    assert scene_indices[0] == frame_slice.start
    assert scene_indices[-1] == frame_slice.stop - 1


@pytest.mark.parametrize("frame_idx", [0, 10, pytest.param(99999, marks=pytest.mark.xfail)])
def test_get_frame_indices_ego(frame_idx: int, zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"
    rasterizer = build_rasterizer(cfg, dmg)
    dataset = EgoDataset(cfg, zarr_dataset, rasterizer)

    frame_indices = dataset.get_frame_indices(frame_idx)
    # this should be only one and match the index of the frame (i.e. it should be frame_idx)
    assert frame_indices[0] == frame_idx


@pytest.mark.parametrize("scene_idx", [0, pytest.param(999, marks=pytest.mark.xfail)])
def test_get_scene_indices_agent(
        scene_idx: int, zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict
) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"
    rasterizer = build_rasterizer(cfg, dmg)
    dataset = AgentDataset(cfg, zarr_dataset, rasterizer)

    # test only first 10 elements
    scene_indices = dataset.get_scene_indices(scene_idx)[:10]
    agents = np.asarray(dataset.dataset.agents)[dataset.agents_mask][:10]

    for agent, idx in zip(agents, scene_indices):
        id_agent = dataset[idx]["track_id"]
        assert id_agent == agent["track_id"]


@pytest.mark.parametrize("frame_idx", [0, 10, 100, 200, pytest.param(999, marks=pytest.mark.xfail)])
def test_get_frame_indices_agent(
        frame_idx: int, zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict
) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"
    rasterizer = build_rasterizer(cfg, dmg)
    dataset = AgentDataset(cfg, zarr_dataset, rasterizer)

    frame_indices = dataset.get_frame_indices(frame_idx)
    # get valid agents from that frame only
    agent_slice = get_agents_slice_from_frames(dataset.dataset.frames[frame_idx])
    agents = dataset.dataset.agents[agent_slice]
    agents = agents[dataset.agents_mask[agent_slice]]

    for agent, idx in zip(agents, frame_indices):
        id_agent = dataset[idx]["track_id"]
        assert id_agent == agent["track_id"]
