from typing import Tuple

import numpy as np
from shapely.geometry import LineString, Polygon


def _get_boundingbox(centroid: np.ndarray, yaw: float, extent: np.ndarray) -> Polygon:
    x, y = centroid[0], centroid[1]
    sin, cos = np.sin(yaw), np.cos(yaw)
    width, length = extent[0] / 2, extent[1] / 2

    x1, y1 = (x + width * cos - length * sin, y + width * sin + length * cos)
    x2, y2 = (x + width * cos + length * sin, y + width * sin - length * cos)
    x3, y3 = (x - width * cos + length * sin, y - width * sin - length * cos)
    x4, y4 = (x - width * cos - length * sin, y - width * sin + length * cos)
    return Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


def _get_sides(bbox: Polygon) -> Tuple[LineString, LineString, LineString, LineString]:
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.exterior.coords[:-1]
    return (
        LineString([(x1, y1), (x2, y2)]),
        LineString([(x3, y3), (x4, y4)]),
        LineString([(x1, y1), (x4, y4)]),
        LineString([(x2, y2), (x3, y3)]),
    )


def within_range(ego_centroid: np.ndarray, ego_extent: np.ndarray, agents: np.ndarray) -> np.ndarray:
    agent_centroids = agents["centroid"]
    agent_extents = agents["extent"]
    distance = np.linalg.norm(ego_centroid - agent_centroids, axis=-1)
    max_range = 0.5 * (np.linalg.norm(ego_extent[:2]) + np.linalg.norm(agent_extents[:, 2], axis=-1))
    return agents[distance < max_range]


def detect_collision(
    pred_centroid: np.ndarray, pred_yaw: float, pred_extent: np.ndarray, target_agents: np.ndarray
) -> Tuple[str, str]:
    """
    Computes whether a collision occurred between ego and any another agent.
    Also computes the type of collision: rear, front, or side.
    For this, we compute the intersection of ego's four sides with a target
    agent and measure the length of this intersection. A collision
    is classified into a class, if the corresponding length is maximal,
    i.e. a front collision exhibits the longest intersection with
    egos front edge.
    """
    ego_bbox = _get_boundingbox(centroid=pred_centroid, yaw=pred_yaw, extent=pred_extent)
    for agent in within_range(pred_centroid, pred_extent, target_agents):
        agent_bbox = _get_boundingbox(agent["centroid"], agent["yaw"], agent["extent"])

        if ego_bbox.intersects(agent_bbox):
            front_side, rear_side, left_side, right_side = _get_sides(ego_bbox)

            intersection_length_per_side = np.asarray(
                [
                    agent_bbox.intersection(front_side).length,
                    agent_bbox.intersection(rear_side).length,
                    agent_bbox.intersection(left_side).length,
                    agent_bbox.intersection(right_side).length,
                ]
            )
            collision_type = ["front", "rear", "side", "side"][np.argmax(intersection_length_per_side)]
            return collision_type, agent["track_id"]
    return "", ""
