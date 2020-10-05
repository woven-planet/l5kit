from typing import Sequence, Union, cast

import numpy as np
import pymap3d as pm
import transforms3d


def compute_agent_pose(agent_centroid_m: np.ndarray, agent_yaw_rad: float) -> np.ndarray:
    """
    Return the agent pose as a 3x3 matrix. This corresponds to world_from_agent matrix.

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
    """
    Return a new matrix that also performs a flip on the y axis.

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


def transform_points_batch(points_batch: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Transform a batch of points using the transformation matrix. This is the batched version of transform_points.
    Note this function assumes points.shape[2] == matrix.shape[1] - 1, which means that the last row on the matrix
    does not influence the final result.
    For 2D points only the first 2x3 part of the matrix will be used.

    Args:
        points_batch (np.ndarray): Input points (BxNx2) or (BxNx3), where B is batch size, N is number of points in the
                                    trajectory.
        transf_matrix (np.ndarray): 3x3 or 4x4 transformation matrix for 2D and 3D input respectively.

    Returns:
        np.ndarray: array of shape (N,2) for 2D input points, or (N,3) points for 3D input points
    """
    assert len(points_batch.shape) == 3, f"points_batch ({points_batch.shape}) must be in the shape of BxNx2 or BxNx3"
    assert len(points_batch.shape[1:]) == len(transf_matrix.shape) == 2, (
        f"dimension mismatch, both points ({points_batch.shape[1:]}) and "
        f"transf_matrix ({transf_matrix.shape}) needs to be a 2D numpy ndarray."
    )
    assert (
        transf_matrix.shape[0] == transf_matrix.shape[1]
    ), f"transf_matrix ({transf_matrix.shape}) should be a transformation matrix."

    if points_batch.shape[2] not in [2, 3]:
        raise AssertionError(
            "Points input should be (N, 2) or (N,3) shape, received {}".format(points_batch.shape[1:])
        )

    assert points_batch.shape[2] == transf_matrix.shape[1] - 1, "points dim should be one less than matrix dim"

    num_dims = len(transf_matrix) - 1
    transf_matrix = transf_matrix.T

    return points_batch @ transf_matrix[:num_dims, :num_dims] + transf_matrix[-1, :num_dims]


def transform_points(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points using transformation matrix. For a batched version, see transform_points_batch.
    Note this function assumes points.shape[1] == matrix.shape[1] - 1, which means that the last
    row on the matrix does not influence the final result.
    For 2D points only the first 2x3 part of the matrix will be used.

    Args:
        points (np.ndarray): Input points (Nx2) or (Nx3).
        transf_matrix (np.ndarray): 3x3 or 4x4 transformation matrix for 2D and 3D input respectively

    Returns:
        np.ndarray: array of shape (N,2) for 2D input points, or (N,3) points for 3D input points
    """
    points_batch = points[np.newaxis, :]
    result = transform_points_batch(points_batch=points_batch, transf_matrix=transf_matrix)
    assert result.shape[0] == 1, f"returned result ({result.shape}) should have shape[0] == 1"
    return result[0, :]


def transform_point(point: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """ Transform a single vector using transformation matrix.

    Args:
        point (np.ndarray): vector of shape (N)
        transf_matrix (np.ndarray): transformation matrix of shape (N+1, N+1)

    Returns:
        np.ndarray: vector of same shape as input point
    """
    point_ext = np.hstack((point, np.ones(1)))
    return np.matmul(transf_matrix, point_ext)[: point.shape[0]]


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
