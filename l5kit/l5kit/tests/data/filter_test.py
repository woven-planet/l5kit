import numpy as np
import pytest

from l5kit.data import (ChunkedDataset, filter_agents_by_frames, filter_agents_by_track_id,
                        get_agents_slice_from_frames, get_frames_slice_from_scenes, get_tl_faces_slice_from_frames)


@pytest.mark.parametrize("frame_bound", [0, 10, 50, 100])
def test_get_frames_agents_shape(frame_bound: int, zarr_dataset: ChunkedDataset) -> None:
    agents = filter_agents_by_frames(zarr_dataset.frames[0:frame_bound], zarr_dataset.agents)
    assert len(agents) == frame_bound


@pytest.mark.parametrize("frame_bound", [pytest.param(0, marks=pytest.mark.xfail), 10, 50, 100])
def test_get_frames_agents_ret(frame_bound: int, zarr_dataset: ChunkedDataset) -> None:
    agents = filter_agents_by_frames(zarr_dataset.frames[0:frame_bound], zarr_dataset.agents)
    assert sum([len(agents_fr) for agents_fr in agents]) > 0


@pytest.mark.parametrize("frame_idx", [0, 10, 50, 100])
def test_get_frames_agents_single_frame(frame_idx: int, zarr_dataset: ChunkedDataset) -> None:
    agents = filter_agents_by_frames(zarr_dataset.frames[frame_idx], zarr_dataset.agents)
    assert len(agents) == 1
    assert isinstance(agents[0], np.ndarray)


def test_get_frames_slice_from_scenes(zarr_dataset: ChunkedDataset) -> None:
    scene_a = zarr_dataset.scenes[0]
    frame_slice = get_frames_slice_from_scenes(scene_a)
    assert len(zarr_dataset.frames) == len(zarr_dataset.frames[frame_slice])

    # test e2e starting from scene
    frame_range = get_frames_slice_from_scenes(zarr_dataset.scenes[0])
    agents_range = get_agents_slice_from_frames(*zarr_dataset.frames[frame_range][[0, -1]])
    tl_faces_range = get_tl_faces_slice_from_frames(*zarr_dataset.frames[frame_range][[0, -1]])
    agents = zarr_dataset.agents[agents_range]
    tl_faces = zarr_dataset.tl_faces[tl_faces_range]

    assert len(agents) == len(zarr_dataset.agents)
    assert len(tl_faces) == len(zarr_dataset.tl_faces)


@pytest.mark.parametrize("slice_end", [1, 10, 20])
def test_get_agents_slice_from_frames(slice_end: int, zarr_dataset: ChunkedDataset) -> None:
    # get agents for first N using function
    frame_slice = slice(0, slice_end)
    agent_slice = get_agents_slice_from_frames(*zarr_dataset.frames[frame_slice][[0, -1]])
    agents_new = zarr_dataset.agents[agent_slice]

    # get agents for first N using standard approach
    frames = zarr_dataset.frames[frame_slice]
    frame_a = frames[0]
    frame_b = frames[-1]
    agents = zarr_dataset.agents[frame_a["agent_index_interval"][0]: frame_b["agent_index_interval"][1]]
    assert np.all(agents_new == agents)


@pytest.mark.parametrize("slice_end", [1, 10, 20])
def test_get_tl_faces_slice_from_frames(slice_end: int, zarr_dataset: ChunkedDataset) -> None:
    # get agents for first N using function
    frame_slice = slice(0, slice_end)
    tl_slice = get_tl_faces_slice_from_frames(*zarr_dataset.frames[frame_slice][[0, -1]])
    tl_faces_new = zarr_dataset.tl_faces[tl_slice]

    # get agents for first N using standard approach
    frames = zarr_dataset.frames[frame_slice]
    frame_a = frames[0]
    frame_b = frames[-1]
    tl_faces = zarr_dataset.tl_faces[
        frame_a["traffic_light_faces_index_interval"][0]: frame_b["traffic_light_faces_index_interval"][1]
    ]
    assert np.all(tl_faces_new == tl_faces)


@pytest.mark.parametrize("track_id", [1, 2, 10, -100])
def test_filter_agents_by_track_id(zarr_dataset: ChunkedDataset, track_id: int) -> None:
    agents = np.asarray(zarr_dataset.agents)
    agents_filtered = filter_agents_by_track_id(agents, track_id)

    # standard approach, iterate through agents and check condition
    agents_filtered_slow = []
    for agent in agents:
        if agent["track_id"] == track_id:
            agents_filtered_slow.append(agent)
    # ensure empty case works for both
    assert len(agents_filtered) == len(agents_filtered_slow)

    if len(agents_filtered) > 0:
        agents_filtered_slow = np.stack(agents_filtered_slow, 0)
        # ensure the elements are the same
        assert np.all(agents_filtered_slow == agents_filtered)


def test_filter_agents_by_track_id_fail_zarr(zarr_dataset: ChunkedDataset) -> None:
    with pytest.raises(IndexError):
        filter_agents_by_track_id(zarr_dataset.agents, 1)  # zarr can't handle boolean indexing
