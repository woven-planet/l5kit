from typing import Tuple

import numpy as np
from shapely.geometry import LineString, Polygon


# TODO(perone):  The functions _ego_agent_within_range, _get_bounding_box,
# _get_sides would ideally be moved to an abstraction of the Agent
# later. Such as in the example below:
#
#  ego = Agent(...)
#  agent = Agent(...)
#  bbox = ego.get_bounding_box()
#  within_range = ego.within_range(agent)
#  sides = ego.get_sides()


def _get_bounding_box(centroid: np.ndarray, yaw: np.ndarray,
                      extent: np.ndarray,) -> Polygon:
    """This function will get a shapely Polygon representing the bounding box
    with an optional buffer around it.

    :param centroid: centroid of the agent
    :param yaw: the yaw of the agent
    :param extent: the extent of the agent
    :return: a shapely Polygon
    """
    x, y = centroid[0], centroid[1]
    sin, cos = np.sin(yaw), np.cos(yaw)
    width, length = extent[0] / 2, extent[1] / 2

    x1, y1 = (x + width * cos - length * sin, y + width * sin + length * cos)
    x2, y2 = (x + width * cos + length * sin, y + width * sin - length * cos)
    x3, y3 = (x - width * cos + length * sin, y - width * sin - length * cos)
    x4, y4 = (x - width * cos - length * sin, y - width * sin + length * cos)
    return Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


# TODO(perone): this should probably return a namedtuple as otherwise it
# would have to depend on the correct ordering of the front/rear/left/right
def _get_sides(bbox: Polygon) -> Tuple[LineString, LineString, LineString, LineString]:
    """This function will get the sides of a bounding box.

    :param bbox: the bounding box
    :return: a tuple with the four sides of the bounding box as LineString,
             representing front/rear/left/right.
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.exterior.coords[:-1]
    return (
        LineString([(x1, y1), (x2, y2)]),
        LineString([(x3, y3), (x4, y4)]),
        LineString([(x1, y1), (x4, y4)]),
        LineString([(x2, y2), (x3, y3)]),
    )


def within_range(ego_centroid: np.ndarray, ego_extent: np.ndarray,
                 agent_centroid: np.ndarray, agent_extent: np.ndarray) -> np.ndarray:
    """This function will check if the agent is within range of the ego. It accepts
    as input a vectorized form with shapes N,D or a flat vector as well with shapes just D.

    :param ego_centroid: the ego centroid (shape: 2)
    :param ego_extent: the ego extent (shape: 3)
    :param agent_centroid: the agent centroid (shape: N, 2)
    :param agent_extent: the agent extent (shape: N, 3)
    :return: array with True if within range, False otherwise (shape: N)
    """
    distance = np.linalg.norm(ego_centroid - agent_centroid, axis=-1)
    norm_ego = np.linalg.norm(ego_extent[:2])
    norm_agent = np.linalg.norm(agent_extent[:, :2], axis=-1)
    max_range = 0.5 * (norm_ego + norm_agent)
    return distance < max_range
