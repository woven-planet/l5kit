from typing import Sequence, Union, cast

import numpy as np
import pymap3d as pm
import transforms3d


def compute_agent_pose(agent_centroid_m: np.ndarray, agent_yaw_rad: float) -> np.ndarray:
    """Return the agent pose as a 3x3 matrix. This corresponds to world_from_agent matrix.

    Args:
        agent_centroid_m (np.ndarry): 2D coordinates of the agent
        agent_yaw_rad (float): yaw of the agent

    Returns:
        (np.ndarray): 3x3 world_from_agent matrix
    """
    # Compute agent pose from its position and heading
    return np.array(
        [
            [np.cos(agent_yaw_rad), -np.sin(agent_yaw_rad), agent_centroid_m[0]],
            [np.sin(agent_yaw_rad), np.cos(agent_yaw_rad), agent_centroid_m[1]],
            [0, 0, 1],
        ]
    )


def rotation33_as_yaw(rotation: np.ndarray) -> float:
    """Compute the yaw component of given 3x3 rotation matrix.

    Args:
        rotation (np.ndarray): 3x3 rotation matrix (np.float64 dtype recommended)

    Returns:
        float: yaw rotation in radians
    """
    return cast(float, transforms3d.euler.mat2euler(rotation)[2])


def yaw_as_rotation33(yaw: float) -> np.ndarray:
    """Create a 3x3 rotation matrix from given yaw.
    The rotation is counter-clockwise and it is equivalent to:
    [cos(yaw), -sin(yaw), 0.0],
    [sin(yaw), cos(yaw), 0.0],
    [0.0, 0.0, 1.0],

    Args:
        yaw (float): yaw rotation in radians

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    return transforms3d.euler.euler2mat(0, 0, yaw)


def flip_y_axis(tm: np.ndarray, y_dim_size: int) -> np.ndarray:
    """Return a new matrix that also performs a flip on the y axis.

    Args:
        tm: the original 3x3 matrix
        y_dim_size: this should match the resolution on y. It makes all coordinates positive

    Returns: a new 3x3 matrix.

    """
    flip_y = np.eye(3)
    flip_y[1, 1] = -1
    tm = np.matmul(flip_y, tm)
    tm[1, 2] += y_dim_size
    return tm


def transform_points(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points using transformation matrices with 3 modes:
    - points (N, f), transf_matrix (f+1, f+1)
        all points are transformed using the matrix and output has shape (N,f).
    - points (B, N, f), transf_matrix (f+1, f+1)
        all sequences of points are transformed using the same matrix and output has shape (B, N,f).
        transf_matrix is broadcasted.
    - points (B, N, f), transf_matrix (B, f+1, f+1)
        each sequence of points is transformed using its own matrix and output has shape (B, N,f).

    Note this function assumes points.shape[-1] == matrix.shape[-1] - 1, which means that last
    rows in the matrices do not influence the final results.
    For 2D points only the first 2x3 parts of the matrices will be used.

    Args:
        points (np.ndarray): Input points (N, 2), (N, 3, (BxNx2), (BxNx3).
        transf_matrix (np.ndarray): Transformation matrix (3x3), (4x4), (Bx3x3), (Bx4x4).

    Returns:
        np.ndarray: transformed points, same shape as input
    """
    assert len(points.shape) in [2, 3], "points should have 2 or 3 dimensions"
    assert len(transf_matrix.shape) in [2, 3], "matrix should have 2 or 3 dimensions"
    assert len(points.shape) >= len(transf_matrix.shape), "points and matrix must have same dim or points one more"

    assert points.shape[-1] in [2, 3], f"last points dimension must be 2 or 3, received {points.shape}"
    assert transf_matrix.shape[-1] in [3, 4], f"last matrix dimension must be 3 or 4 received {transf_matrix.shape}"
    assert (
        transf_matrix.shape[-1] == transf_matrix.shape[-2]
    ), f"transf_matrix ({transf_matrix.shape}) should be a square matrix."
    assert points.shape[-1] == transf_matrix.shape[-1] - 1, "points last dim should be one less than matrix's one"

    def _transform(points: np.ndarray, transf_matrix: np.ndarray):
        num_dims = transf_matrix.shape[-1] - 1
        transf_matrix = np.transpose(transf_matrix, (0, 2, 1))
        return points @ transf_matrix[:, :num_dims, :num_dims] + transf_matrix[:, -1:, :num_dims]

    if len(points.shape) == len(transf_matrix.shape) == 2:
        points = np.expand_dims(points, 0)
        transf_matrix = np.expand_dims(transf_matrix, 0)
        return _transform(points, transf_matrix)[0]

    elif len(points.shape) == len(transf_matrix.shape) == 3:
        return _transform(points, transf_matrix)

    elif len(points.shape) == 3 and len(transf_matrix.shape) == 2:
        transf_matrix = np.expand_dims(transf_matrix, 0)
        return _transform(points, transf_matrix)
    else:
        raise NotImplementedError(f"unhandled case! points: {points.shape}, matrix: {transf_matrix.shape}")


def transform_point(point: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """Transform a single vector using transformation matrix.
        This function call transform_points internally
    Args:
        point (np.ndarray): vector of shape (N)
        transf_matrix (np.ndarray): transformation matrix of shape (N+1, N+1)

    Returns:
        np.ndarray: vector of same shape as input point
    """
    point = np.expand_dims(point, 0)
    return transform_points(point, transf_matrix)[0]


def ecef_to_geodetic(point: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert given ECEF coordinate into latitude, longitude, altitude.

    Args:
        point (Union[np.ndarray, Sequence[float]]): ECEF coordinate vector

    Returns:
        np.ndarray: latitude, altitude, longitude
    """
    return np.array(pm.ecef2geodetic(point[0], point[1], point[2]))


def geodetic_to_ecef(lla_point: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert given latitude, longitude, and optionally altitude into ECEF
    coordinates. If no altitude is given, altitude 0 is assumed.

    Args:
        lla_point (Union[np.ndarray, Sequence[float]]): Latitude, Longitude and optionally Altitude

    Returns:
        np.ndarray: 3D ECEF coordinate
    """
    if len(lla_point) == 2:
        return np.array(pm.geodetic2ecef(lla_point[0], lla_point[1], 0), dtype=np.float64)
    else:
        return np.array(pm.geodetic2ecef(lla_point[0], lla_point[1], lla_point[2]), dtype=np.float64)
