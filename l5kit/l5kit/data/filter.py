from typing import List, Optional

import numpy as np

from .labels import LABEL_TO_INDEX

# Labels that belong to "agents" of some sort.
LABELS_TO_KEEP = [
    "PERCEPTION_LABEL_CAR",
    "PERCEPTION_LABEL_VAN",
    "PERCEPTION_LABEL_TRAM",
    "PERCEPTION_LABEL_BUS",
    "PERCEPTION_LABEL_TRUCK",
    "PERCEPTION_LABEL_EMERGENCY_VEHICLE",
    "PERCEPTION_LABEL_OTHER_VEHICLE",
    "PERCEPTION_LABEL_BICYCLE",
    "PERCEPTION_LABEL_MOTORCYCLE",
    "PERCEPTION_LABEL_CYCLIST",
    "PERCEPTION_LABEL_MOTORCYCLIST",
    "PERCEPTION_LABEL_PEDESTRIAN",
    "PERCEPTION_LABEL_ANIMAL",
]
LABEL_INDICES_TO_KEEP = [LABEL_TO_INDEX[label] for label in LABELS_TO_KEEP]


def _get_label_filter(label_probabilities: np.ndarray, threshold: float) -> np.array:
    """

    Arguments:
        label_probabilities (np.ndarray): Given the probabilities of all labels, returns a binary mask
        of those whose summed probability of the classes we are interested in is higher than the given threshold.

        This set of classes "we are interested in" is hardcoded for now.

    Keyword Arguments:
        threshold (float): probability threshold for filtering

    Returns:
        np.array -- A binary array which can be used to mask agents.
    """
    return np.sum(label_probabilities[:, LABEL_INDICES_TO_KEEP], axis=1) > threshold


def filter_agents_by_labels(agents: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Filters an agents array, keeping those agents that meet the threshold.

    Arguments:
        agents (np.ndarray): Agents array

    Keyword Arguments:
        threshold (float): probability threshold for filtering (default: {0.5})

    Returns:
        np.ndarray -- A subset of input ``agents`` array.
    """
    label_indices = _get_label_filter(agents["label_probabilities"], threshold)
    return agents[label_indices]


def filter_agents_by_track_id(agents: np.ndarray, track_id: int) -> np.ndarray:
    """Return all agent object (np.ndarray) of a given track_id.

    Arguments:
        agents (np.ndarray): agents array
        track_id (int): agent track id to select

    Returns:
        np.ndarray -- Selected agent.
    """
    return agents[np.nonzero(agents["track_id"] == track_id)[0]]


def get_agent_by_track_id(agents_frame: np.ndarray, track_id: int) -> Optional[np.ndarray]:
    """Return the agent object (np.ndarray) of a given track_id in a frame.
    Return None if the agent is not among those in the frame.

    Arguments:
        agents_frame (np.ndarray): frame agents array
        track_id (int): agent track id to select

    Returns:
        Optional[np.ndarray] -- Selected agent, or None if this agent is not present in given frame.
    """

    try:
        agent = filter_agents_by_track_id(agents_frame, track_id)[0]
        return agent
    except IndexError:  # no agent for track_id in this frame
        return None


def filter_agents_by_frames(frames: np.ndarray, agents: np.ndarray) -> List[np.ndarray]:
    """
    Get a list of agents array, one array per frame. Note that "agent_index_interval" is used to filter agents,
    so you should take care of re-setting it if you have previously sliced agents.

    Args:
        frames (np.ndarray): an array of frames
        agents (np.ndarray): an array of agents

    Returns:
        List[np.ndarray] with the agents divided by frame
    """
    assert frames.shape != (), "an array of frames is required, use frames[None] or un-squeeze"
    return [agents[slice(*frame["agent_index_interval"])] for frame in frames]
