import numpy as np
from l5kit.data import get_agents_slice_from_frames, ChunkedDataset


def insert_agent(agent: np.ndarray, frame_idx: int, dataset: ChunkedDataset):
    """Insert an agent in one frame.
    Assumptions:
        the dataset has only 1 scene
        the dataset is in numpy format and not zarr anymore

    :param agent: the agent info to be inserted
    :param frame_idx: the frame where we want to insert the agent
    :param dataset: the single-scene dataset.
    """
    if not len(dataset.scenes) == 1:
        raise ValueError(f"dataset should have a single scene, got {len(dataset.scenes)}")
    if not isinstance(dataset.agents, np.ndarray):
        raise ValueError("dataset agents should be an editable np array")
    if not isinstance(dataset.frames, np.ndarray):
        raise ValueError("dataset frames should be an editable np array")
    if not frame_idx < len(dataset.frames):
        raise ValueError(f"can't set frame {frame_idx} in dataset with len {len(dataset.frames)}")

    frame = dataset.frames[frame_idx]
    agents_slice = get_agents_slice_from_frames(frame)
    agents_frame = dataset.agents[agents_slice]

    idx_set = np.argwhere(agent["track_id"] == agents_frame["track_id"])
    assert len(idx_set) in [0, 1]

    if len(idx_set):
        # CASE 1
        # the agent is already there and we can just update it
        # we set also label_probabilities from the current one to ensure it is high enough
        idx_set = int(idx_set[0])
        agents_frame[idx_set: idx_set + 1] = agent
    else:
        # CASE 2
        # we need to insert the agent and move everything
        dataset.agents = np.concatenate(
            [dataset.agents[0: agents_slice.stop], agent, dataset.agents[agents_slice.stop:]], 0
        )

        # move end of the current frame and all other frames start and end
        dataset.frames[frame_idx]["agent_index_interval"] += (0, 1)
        dataset.frames[frame_idx + 1:]["agent_index_interval"] += 1


def disable_agents(dataset: ChunkedDataset, allowlist: np.ndarray):
    """Disable all agents in dataset except for the ones in allowlist
    Assumptions:
        the dataset has only 1 scene
        the dataset is in numpy format and not zarr anymore

    :param dataset: the single-scene dataset
    :param allowlist: 1D np array of track_ids to keep

    """
    if not len(dataset.scenes) == 1:
        raise ValueError(f"dataset should have a single scene, got {len(dataset.scenes)}")
    if not isinstance(dataset.agents, np.ndarray):
        raise ValueError("dataset agents should be an editable np array")
    if not len(allowlist.shape) == 1:
        raise ValueError("allow list should be 1D")

    agent_track_ids = dataset.agents["track_id"]

    mask_disable = ~np.in1d(agent_track_ids, allowlist)

    # this will set those agents as invisible
    # we also zeroes their pose and extent
    dataset.agents["centroid"][mask_disable] *= 0
    dataset.agents["yaw"][mask_disable] *= 0
    dataset.agents["extent"][mask_disable] *= 0
    dataset.agents["label_probabilities"][mask_disable] = -1
