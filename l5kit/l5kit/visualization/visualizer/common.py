from typing import List, NamedTuple

import numpy as np


class MapElementVisualization(NamedTuple):
    """Hold information about a single element to be visualised on the map

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param color: color of the lane as a string (both hex or text)
    :param alpha: [0, 1] value of transparency

    """
    xs: np.ndarray
    ys: np.ndarray
    color: str
    alpha: float


class AgentVisualization(NamedTuple):
    """Hold information about a single agent

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param color: color of the agent as a string (both hex or text)
    :param alpha: [0, 1] value of transparency
    :param track_id: track id of the agent (unique in a scene)
    :param agent_type: type of the agent as a string (e.g. pedestrian)
    :param prob: probability of the agent from PCB
    """
    xs: np.ndarray
    ys: np.ndarray
    color: str
    alpha: float
    track_id: int
    agent_type: str
    prob: float


class EgoVisualization(NamedTuple):
    """Hold information about a single ego annotation

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param color: color of the ego as a string (both hex or text)
    :param alpha: [0, 1] value of transparency
    :param center_x: the center x coordinate of the ego bbox
    :param center_y: the center y coordinate of the ego bbox
    """
    xs: np.ndarray
    ys: np.ndarray
    color: str
    alpha: float
    center_x: float
    center_y: float


class TrajectoryVisualization(NamedTuple):
    """Hold information about a single trajectory annotation

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param color: color of the lane as a string (both hex or text)
    :param legend_label: the name of this trajectory for the legend (e.g. `ego_trajectory`)
    :param track_id: the track id of the associated agent
    """
    xs: np.ndarray
    ys: np.ndarray
    color: str
    legend_label: str
    track_id: int


class FrameVisualization(NamedTuple):
    """Hold information about a frame (the state of a scene at a given time)

    :param ego: a list of ego annotations. Usually this list has only one element inside
    :param agents: a list of agents
    :param map_patches: a list of patch for the map
    :param map_lines: a list of lines for the map
    :param trajectories: a list of trajectories
    """
    ego: List[EgoVisualization]
    agents: List[AgentVisualization]
    map_patches: List[MapElementVisualization]
    map_lines: List[MapElementVisualization]
    trajectories: List[TrajectoryVisualization]
