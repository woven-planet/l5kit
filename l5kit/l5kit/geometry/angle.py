from typing import cast

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """angle_between_vectors returns the angle in radians between two vectors.

    Args:
        v1 (np.ndarray): Vector 1 of shape (N)
        v2 (np.ndarray): Vector 2 of same shape as ``v1``

    Returns:
        float: angle in radians
    """
    cos_ang = np.dot(v1, v2)
    sin_ang = np.linalg.norm(np.cross(v1, v2))
    return cast(float, np.arctan2(sin_ang, cos_ang))


def compute_yaw_around_north_from_direction(direction_vector: np.ndarray) -> float:
    """compute_yaw_from_direction computes the yaw as angle between a 2D input direction vector and
the y-axis direction vector (0, 1).

    Args:
        direction_vector (np.ndarray): Vector of shape (2,)

    Returns:
        float: angle to (0,1) vector in radians
    """
    return angle_between_vectors(direction_vector, np.array([0.0, 1.0]))
