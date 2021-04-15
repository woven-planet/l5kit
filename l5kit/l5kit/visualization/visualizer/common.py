from typing import List, NamedTuple

import numpy as np


class LaneVisualization(NamedTuple):
    """Hold information about a single lane

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param color: color of the lane as a string (both hex or text)
    """
    xs: np.ndarray
    ys: np.ndarray
    color: str


class CWVisualization(NamedTuple):
    """Hold information about a single crosswalk

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param color: color of the lane as a string (both hex or text)
    """
    xs: np.ndarray
    ys: np.ndarray
    color: str


class AgentVisualization(NamedTuple):
    """Hold information about a single agent

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param color: color of the lane as a string (both hex or text)
    :param track_id: track id of the agent (unique in a scene)
    :param agent_type: type of the agent as a string (e.g. pedestrian)
    :param prob: probability of the agent from PCB
    """
    xs: np.ndarray
    ys: np.ndarray
    color: str
    track_id: int
    agent_type: str
    prob: float


class EgoVisualization(NamedTuple):
    """Hold information about a single ego annotation

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param color: color of the lane as a string (both hex or text)
    :param center_x: the center x coordinate of the ego bbox
    :param center_y: the center y coordinate of the ego bbox
    """
    xs: np.ndarray
    ys: np.ndarray
    color: str
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

    :param ego: a single ego annotation
    :param agents: a list of agents
    :param lanes: a list of lanes
    :param crosswalks: a list of crosswalks
    :param trajectories: a list of trajectories
    """
    ego: EgoVisualization
    agents: List[AgentVisualization]
    lanes: List[LaneVisualization]
    crosswalks: List[CWVisualization]
    trajectories: List[TrajectoryVisualization]
