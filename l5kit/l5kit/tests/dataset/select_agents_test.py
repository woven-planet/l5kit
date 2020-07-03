from l5kit.data import ChunkedStateDataset
from l5kit.dataset.select_agents import get_valid_agents, TH_DISTANCE_AV, TH_MOVEMENT, TH_EXTENT_RATIO, TH_YAW_DEGREE
import numpy as np
import pytest
from functools import partial

SCENE_LENGTH = 50

get_valid_agents_p = partial(get_valid_agents, th_agent_filter_probability_threshold=0,
                     th_yaw_degree=TH_YAW_DEGREE, th_extent_ratio=TH_EXTENT_RATIO,
                     th_movement=TH_MOVEMENT, th_distance_av=TH_DISTANCE_AV)


@pytest.fixture()  # not shared in scope
def dataset() -> ChunkedStateDataset:
    dataset = ChunkedStateDataset("")
    dataset.scenes = np.zeros(1, dtype=dataset.scenes.dtype)
    dataset.frames = np.zeros(SCENE_LENGTH, dtype=dataset.frames.dtype)
    dataset.agents = np.zeros(SCENE_LENGTH, dtype=dataset.agents.dtype)

    dataset.scenes[0]["frame_index_interval"] = (0, SCENE_LENGTH)
    for idx in range(len(dataset.frames)):
        dataset.frames[idx]["agent_index_interval"] = (idx, idx + 1)

    for idx in range(len(dataset.agents)):
        dataset.agents[idx]["centroid"] = (idx, idx * 0.1)  # ensure we don't go too far from the AV
        dataset.agents[idx]["extent"] = (5, 5, 5)
        dataset.agents[idx]["yaw"] = 0
        dataset.agents[idx]["track_id"] = 1
        dataset.agents[idx]["label_probabilities"][3] = 1.0

    return dataset


def test_get_valid_agents_annot_hole(dataset: ChunkedStateDataset):
    frames_range = np.asarray([0, len(dataset.frames)])
    # put an annotation hole at 10
    dataset.agents[10]["track_id"] = 2

    agents_mask, *_ = get_valid_agents_p(frames_range, dataset)
    # 9 should have no future and 11 no past
    assert agents_mask[9, 1] == 0
    assert agents_mask[11, 0] == 0
    # 10 shouldn't have no past nor future
    assert agents_mask[10, 0] == agents_mask[10, 1] == 0


def test_get_valid_agents_multi_annot_hole(dataset: ChunkedStateDataset):
    frames_range = np.asarray([0, len(dataset.frames)])
    # put an annotation hole at 10 and 25
    dataset.agents[10]["track_id"] = 2
    dataset.agents[25]["track_id"] = 2

    agents_mask, *_ = get_valid_agents_p(frames_range, dataset)

    assert np.all((agents_mask[1:10, 0] - agents_mask[:9, 0]) == 1)
    assert np.all((agents_mask[1:10, 1] - agents_mask[:9, 1]) == -1)
    assert agents_mask[10, 0] == agents_mask[10, 1] == 0

    assert np.all((agents_mask[12:25, 0] - agents_mask[11:24, 0]) == 1)
    assert np.all((agents_mask[12:25, 1] - agents_mask[11:24, 1]) == -1)
    assert agents_mask[25, 0] == agents_mask[25, 1] == 0


def test_get_valid_agents_centroid_change(dataset: ChunkedStateDataset):
    frames_range = np.asarray([0, len(dataset.frames)])
    # change centroid
    dataset.agents[10]["centroid"] *= 2

    agents_mask, *_ = get_valid_agents_p(frames_range, dataset)

    assert np.all((agents_mask[1:10, 0] - agents_mask[:9, 0]) == 1)
    assert np.all((agents_mask[1:10, 1] - agents_mask[:9, 1]) == -1)
    assert agents_mask[10, 0] == agents_mask[10, 1] == 0


def test_get_valid_agents(dataset: ChunkedStateDataset):
    frames_range = np.asarray([0, len(dataset.frames)])
    agents_mask, *_ = get_valid_agents_p(frames_range, dataset)

    # we have a single valid agents, so the mask should decrease gently in the future and increase in the past
    assert np.all((agents_mask[1:, 0] - agents_mask[:-1, 0]) == 1)
    assert np.all((agents_mask[1:, 1] - agents_mask[:-1, 1]) == -1)
