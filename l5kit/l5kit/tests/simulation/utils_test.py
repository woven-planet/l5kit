import unittest

import numpy as np
import pytest

from l5kit.data import (AGENT_DTYPE, ChunkedDataset, FRAME_DTYPE, get_agents_slice_from_frames,
                        get_tl_faces_slice_from_frames, SCENE_DTYPE, TL_FACE_DTYPE)
from l5kit.simulation.utils import disable_agents, get_frames_subset, insert_agent


class TestAgentInsert(unittest.TestCase):
    def _get_simple_dataset(self) -> ChunkedDataset:
        # build a simple dataset with 3 frames
        # frame 0:
        #   agent 0
        #   agent 1
        #   agent 2
        # frame 1:
        #   agent 0
        #   agent 1
        # frame 2:
        #   agent 0

        dataset = ChunkedDataset("")
        dataset.scenes = np.zeros(1, dtype=SCENE_DTYPE)
        dataset.frames = np.zeros(3, dtype=FRAME_DTYPE)
        dataset.agents = np.zeros(6, dtype=AGENT_DTYPE)

        dataset.scenes[0]["frame_index_interval"] = (0, 3)
        dataset.frames["agent_index_interval"] = [(0, 3), (3, 5), (5, 6)]

        dataset.agents["track_id"] = [0, 1, 2, 0, 1, 0]
        return dataset

    def test_invalid(self) -> None:
        # try to insert out of bounds
        with self.assertRaises(ValueError):
            insert_agent(
                np.zeros(1, dtype=AGENT_DTYPE), 100, self._get_simple_dataset()
            )
        # try to insert in a multi-scene dataset
        with self.assertRaises(ValueError):
            dataset = self._get_simple_dataset()
            dataset.scenes = np.concatenate([dataset.scenes, dataset.scenes])
            insert_agent(np.zeros(1, dtype=AGENT_DTYPE), 100, dataset)

    def test_update(self) -> None:
        # try to update agent 0 in frame 0
        dataset = self._get_simple_dataset()
        agent = np.zeros(1, dtype=AGENT_DTYPE)
        agent["centroid"] = (1, 1)
        agent["yaw"] = 1
        agent["label_probabilities"] += 1

        insert_agent(agent, 0, dataset)

        self.assertTrue(np.allclose(agent["centroid"], dataset.agents[0]["centroid"]))
        self.assertTrue(np.allclose(agent["yaw"], dataset.agents[0]["yaw"]))
        self.assertTrue(
            np.allclose(
                agent["label_probabilities"], dataset.agents[0]["label_probabilities"]
            )
        )

        # the number of agents should be the same
        self.assertTrue(len(dataset.agents), len(self._get_simple_dataset().agents))
        # the boundries should be the same
        self.assertTrue(
            np.allclose(
                dataset.frames["agent_index_interval"],
                self._get_simple_dataset().frames["agent_index_interval"],
            )
        )

    def test_insert_1(self) -> None:
        # try to insert agent 2 in frame 1
        dataset = self._get_simple_dataset()
        agent = np.zeros(1, dtype=AGENT_DTYPE)
        agent["centroid"] = (1, 1)
        agent["yaw"] = 1
        agent["label_probabilities"] += 1
        agent["track_id"] += 2

        insert_agent(agent, 1, dataset)

        # the new agent should be in position 5
        expected_index = 5

        self.assertTrue(
            np.allclose(agent["centroid"], dataset.agents[expected_index]["centroid"])
        )
        self.assertTrue(
            np.allclose(agent["yaw"], dataset.agents[expected_index]["yaw"])
        )
        self.assertTrue(
            np.allclose(
                agent["label_probabilities"],
                dataset.agents[expected_index]["label_probabilities"],
            )
        )

        # the number of agents should be 1 more
        old_dataset = self._get_simple_dataset()
        self.assertTrue(len(dataset.agents), len(old_dataset.agents) + 1)

        # the interval should be the same for frame 0 and 1 start
        self.assertTrue(
            np.allclose(
                dataset.frames["agent_index_interval"][:1],
                old_dataset.frames["agent_index_interval"][:1],
            )
        )
        self.assertTrue(
            np.allclose(
                dataset.frames["agent_index_interval"][1, 0],
                old_dataset.frames["agent_index_interval"][1, 0],
            )
        )

        # from 1 end to end they should be one more
        self.assertTrue(
            np.allclose(
                dataset.frames["agent_index_interval"][1, 1],
                old_dataset.frames["agent_index_interval"][1, 1] + 1,
            )
        )
        self.assertTrue(
            np.allclose(
                dataset.frames["agent_index_interval"][2:],
                old_dataset.frames["agent_index_interval"][2:] + 1,
            )
        )

    def test_insert_2(self) -> None:
        # try to insert agent 1 in frame 2
        dataset = self._get_simple_dataset()
        agent = np.zeros(1, dtype=AGENT_DTYPE)
        agent["centroid"] = (1, 1)
        agent["yaw"] = 1
        agent["label_probabilities"] += 1
        agent["track_id"] += 1

        insert_agent(agent, 2, dataset)

        # the new agent should be the last one
        expected_index = -1

        self.assertTrue(
            np.allclose(agent["centroid"], dataset.agents[expected_index]["centroid"])
        )
        self.assertTrue(
            np.allclose(agent["yaw"], dataset.agents[expected_index]["yaw"])
        )
        self.assertTrue(
            np.allclose(
                agent["label_probabilities"],
                dataset.agents[expected_index]["label_probabilities"],
            )
        )

        # the number of agents should be 1 more
        old_dataset = self._get_simple_dataset()
        self.assertTrue(len(dataset.agents), len(old_dataset.agents) + 1)

        # the interval should be the same for frame 0, 1 and 2 start
        self.assertTrue(
            np.allclose(
                dataset.frames["agent_index_interval"][:-1],
                old_dataset.frames["agent_index_interval"][:-1],
            )
        )
        self.assertTrue(
            np.allclose(
                dataset.frames["agent_index_interval"][-1, 0],
                old_dataset.frames["agent_index_interval"][-1, 0],
            )
        )

        # the very last index should be 1 more
        self.assertTrue(
            np.allclose(
                dataset.frames["agent_index_interval"][-1, 1],
                old_dataset.frames["agent_index_interval"][-1, 1] + 1,
            )
        )


class TestDisableAgents(unittest.TestCase):
    def _get_simple_dataset(self) -> ChunkedDataset:
        # build a simple dataset with 3 frames
        # frame 0:
        #   agent 0
        #   agent 1
        #   agent 2
        # frame 1:
        #   agent 0
        #   agent 1
        # frame 2:
        #   agent 0

        dataset = ChunkedDataset("")
        dataset.scenes = np.zeros(1, dtype=SCENE_DTYPE)
        dataset.frames = np.zeros(3, dtype=FRAME_DTYPE)
        dataset.agents = np.zeros(6, dtype=AGENT_DTYPE)

        dataset.scenes[0]["frame_index_interval"] = (0, 3)
        dataset.frames["agent_index_interval"] = [(0, 3), (3, 5), (5, 6)]

        dataset.agents["track_id"] = [0, 1, 2, 0, 1, 0]
        # set properties to something different than 0
        dataset.agents["centroid"] = np.random.rand(*dataset.agents["centroid"].shape)
        dataset.agents["yaw"] = np.random.rand(*dataset.agents["yaw"].shape)
        dataset.agents["extent"] = np.random.rand(*dataset.agents["extent"].shape)
        return dataset

    def test_invalid(self) -> None:
        # try to delete agents in a multi-scene dataset
        with self.assertRaises(ValueError):
            dataset = self._get_simple_dataset()
            dataset.scenes = np.concatenate([dataset.scenes, dataset.scenes])
            disable_agents(dataset, np.arange(2))

        # try to delete agents using a N-D allowlist
        with self.assertRaises(ValueError):
            dataset = self._get_simple_dataset()
            disable_agents(dataset, np.zeros((2, 2)))

    def test_allowlist(self) -> None:
        dataset = self._get_simple_dataset()
        original_agents = dataset.agents.copy()

        disable_agents(dataset, np.asarray([1]))

        # all agents except the 2 and -2 should be removed
        for agent_idx in [0, 2, 3, -1]:
            self.assertTrue(np.allclose(dataset.agents[agent_idx]["centroid"], 0))
            self.assertTrue(np.allclose(dataset.agents[agent_idx]["yaw"], 0))
            self.assertTrue(np.allclose(dataset.agents[agent_idx]["extent"], 0))
            self.assertTrue(
                np.allclose(dataset.agents[agent_idx]["label_probabilities"], -1)
            )

        # those two should have been left untouched
        for agent_idx in [1, -2]:
            new_agent = dataset.agents[agent_idx]
            old_agent = original_agents[agent_idx]
            self.assertTrue(np.allclose(new_agent["centroid"], old_agent["centroid"]))
            self.assertTrue(np.allclose(new_agent["yaw"], old_agent["yaw"]))
            self.assertTrue(np.allclose(new_agent["extent"], old_agent["extent"]))
            self.assertTrue(
                np.allclose(
                    new_agent["label_probabilities"], old_agent["label_probabilities"]
                )
            )


def test_dataset_frames_subset_invalid(zarr_dataset: ChunkedDataset) -> None:
    with pytest.raises(ValueError):
        get_frames_subset(zarr_dataset, 0, 10)  # not in numpy
    with pytest.raises(ValueError):
        get_frames_subset(zarr_dataset, -1, 10)  # invalid start
    with pytest.raises(ValueError):
        get_frames_subset(
            zarr_dataset, len(zarr_dataset.frames), len(zarr_dataset.frames) + 1
        )  # invalid start
    with pytest.raises(ValueError):
        get_frames_subset(zarr_dataset, 10, 9)  # invalid end
    with pytest.raises(ValueError):
        get_frames_subset(zarr_dataset, 10, len(zarr_dataset.frames) + 1)  # invalid end


def test_dataset_frames_subset(zarr_dataset: ChunkedDataset) -> None:
    zarr_dataset = zarr_dataset.get_scene_dataset(0)
    frame_start = 10
    frame_end = 25
    zarr_cut = get_frames_subset(zarr_dataset, frame_start, frame_end)

    assert len(zarr_cut.scenes) == 1
    assert len(zarr_cut.frames) == frame_end - frame_start
    assert np.all(
        zarr_cut.frames["ego_translation"]
        == zarr_dataset.frames["ego_translation"][frame_start:frame_end]
    )

    agents_slice = get_agents_slice_from_frames(
        *zarr_dataset.frames[[frame_start, frame_end - 1]]
    )
    tls_slice = get_tl_faces_slice_from_frames(
        *zarr_dataset.frames[[frame_start, frame_end - 1]]
    )

    assert np.all(zarr_cut.agents == zarr_dataset.agents[agents_slice])
    assert np.all(zarr_cut.tl_faces == zarr_dataset.tl_faces[tls_slice])
    assert np.all(zarr_cut.scenes["frame_index_interval"] == (0, len(zarr_cut.frames)))


def test_mock_dataset_frames_subset() -> None:
    zarr_dataset = ChunkedDataset("")
    zarr_dataset.scenes = np.zeros(1, dtype=SCENE_DTYPE)
    zarr_dataset.scenes[0]["frame_index_interval"] = (0, 4)
    zarr_dataset.frames = np.zeros(4, dtype=FRAME_DTYPE)
    zarr_dataset.frames["agent_index_interval"] = [(0, 1), (1, 2), (2, 3), (3, 4)]
    zarr_dataset.agents = np.zeros(4, dtype=AGENT_DTYPE)
    zarr_dataset.agents["track_id"] = np.arange(4)
    zarr_dataset.tl_faces = np.zeros(0, dtype=TL_FACE_DTYPE)

    frame_start = 1
    frame_end = 3
    zarr_cut = get_frames_subset(zarr_dataset, frame_start, frame_end)
    assert np.all(zarr_cut.agents["track_id"] == [1, 2])

    frame_start = 0
    frame_end = 3
    zarr_cut = get_frames_subset(zarr_dataset, frame_start, frame_end)
    assert np.all(zarr_cut.agents["track_id"] == [0, 1, 2])

    frame_start = 2
    frame_end = 4
    zarr_cut = get_frames_subset(zarr_dataset, frame_start, frame_end)
    assert np.all(zarr_cut.agents["track_id"] == [2, 3])
