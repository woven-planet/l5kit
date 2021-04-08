from typing import List, NamedTuple

import numpy as np


class LaneVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str


class CWVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str


class AgentVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str
    track_id: int
    type: str
    prob: float


class EgoVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str
    center_x: float
    center_y: float


class TrajectoryVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str
    legend_label: str
    track_id: int


class FrameVisualisation(NamedTuple):
    ego: EgoVisualisation
    agents: List[AgentVisualisation]
    lanes: List[LaneVisualisation]
    crosswalks: List[CWVisualisation]
    trajectories: List[TrajectoryVisualisation]
