from functools import partial
from pathlib import Path

import numpy as np
import pytest

from l5kit.data import ChunkedDataset
from l5kit.dataset.select_agents import get_valid_agents, TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE


SCENE_LENGTH = 50

get_valid_agents_p = partial(
    get_valid_agents,
    th_agent_filter_probability_threshold=0,
    th_yaw_degree=TH_YAW_DEGREE,
    th_extent_ratio=TH_EXTENT_RATIO,
    th_distance_av=TH_DISTANCE_AV,
)


@pytest.fixture()  # not shared in scope
def dataset(tmp_path: Path) -> ChunkedDataset:
    dataset = ChunkedDataset(str(tmp_path))
    dataset.scenes = np.zeros(1, dtype=dataset.scenes.dtype)
    dataset.frames = np.zeros(SCENE_LENGTH, dtype=dataset.frames.dtype)
    dataset.agents = np.zeros(SCENE_LENGTH, dtype=dataset.agents.dtype)

    dataset.scenes[0]["frame_index_interval"] = (0, SCENE_LENGTH)
    for idx in range(len(dataset.frames)):
        dataset.frames[idx]["agent_index_interval"] = (idx, idx + 1)
        dataset.frames[idx]["timestamp"] = idx

    for idx in range(len(dataset.agents)):
        # we don't check moving anymore, so the agent can stay still
        dataset.agents[idx]["extent"] = (5, 5, 5)
        dataset.agents[idx]["yaw"] = 0
        dataset.agents[idx]["track_id"] = 1
        dataset.agents[idx]["label_probabilities"][3] = 1.0

    return dataset


def test_get_valid_agents_annot_hole(dataset: ChunkedDataset) -> None:
    frames_range = np.asarray([0, len(dataset.frames)])
    # put an annotation hole at 10
    dataset.agents[10]["track_id"] = 2

    agents_mask, *_ = get_valid_agents_p(frames_range, dataset)
    agents_mask = agents_mask.astype(np.int)

    # 9 should have no future and 11 no past
    assert agents_mask[9, 1] == 0
    assert agents_mask[11, 0] == 0
    # 10 should have no past nor future
    assert agents_mask[10, 0] == agents_mask[10, 1] == 0

    # past should increase and future decrease until 10, then from 11 to the end
    assert np.all(np.diff(agents_mask[:10, 0]) == 1)
    assert np.all(np.diff(agents_mask[:10, 1]) == -1)
    assert np.all(np.diff(agents_mask[11:, 0]) == 1)
    assert np.all(np.diff(agents_mask[11:, 1]) == -1)


def test_get_valid_agents_multi_annot_hole(dataset: ChunkedDataset) -> None:
    frames_range = np.asarray([0, len(dataset.frames)])
    # put an annotation hole at 10 and 25
    dataset.agents[10]["track_id"] = 2
    dataset.agents[25]["track_id"] = 2

    agents_mask, *_ = get_valid_agents_p(frames_range, dataset)
    agents_mask = agents_mask.astype(np.int)

    assert np.all(np.diff(agents_mask[:10, 0]) == 1)
    assert np.all(np.diff(agents_mask[:10, 1]) == -1)
    assert agents_mask[10, 0] == agents_mask[10, 1] == 0

    assert np.all(np.diff(agents_mask[11:25, 0]) == 1)
    assert np.all(np.diff(agents_mask[11:25, 1]) == -1)
    assert agents_mask[25, 0] == agents_mask[25, 1] == 0


def test_get_valid_agents_extent_change(dataset: ChunkedDataset) -> None:
    frames_range = np.asarray([0, len(dataset.frames)])
    # change centroid
    dataset.agents[10]["extent"] *= 2

    agents_mask, *_ = get_valid_agents_p(frames_range, dataset)
    agents_mask = agents_mask.astype(np.int)

    assert np.all(np.diff(agents_mask[:10, 0]) == 1)
    assert np.all(np.diff(agents_mask[:10, 1]) == -1)
    assert agents_mask[10, 0] == agents_mask[10, 1] == 0


def test_get_valid_agents_yaw_change(dataset: ChunkedDataset) -> None:
    frames_range = np.asarray([0, len(dataset.frames)])
    # change centroid
    dataset.agents[10]["yaw"] = np.radians(50)
    dataset.agents[20]["yaw"] = np.radians(29)  # under yaw threshold

    agents_mask, *_ = get_valid_agents_p(frames_range, dataset)
    agents_mask = agents_mask.astype(np.int)

    assert np.all(np.diff(agents_mask[:10, 0]) == 1)
    assert np.all(np.diff(agents_mask[:10, 1]) == -1)

    assert agents_mask[10, 0] == agents_mask[10, 1] == 0

    assert np.all(np.diff(agents_mask[11:, 0]) == 1)
    assert np.all(np.diff(agents_mask[11:, 1]) == -1)


def test_get_valid_agents(dataset: ChunkedDataset) -> None:
    frames_range = np.asarray([0, len(dataset.frames)])
    agents_mask, *_ = get_valid_agents_p(frames_range, dataset)
    agents_mask = agents_mask.astype(np.int)

    # we have a single valid agents, so the mask should decrease gently in the future and increase in the past
    assert np.all(np.diff(agents_mask[:, 0]) == 1)
    assert np.all(np.diff(agents_mask[:, 1]) == -1)
