from typing import cast, Sequence, Union

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


def vertical_flip(tm: np.ndarray, y_dim_size: int) -> np.ndarray:
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
    Transform a set of 2D/3D points using the given transformation matrix.
    Assumes row major ordering of the input points. The transform function has 3 modes:
    - points (N, F), transf_matrix (F+1, F+1)
    all points are transformed using the matrix and the output points have shape (N, F).
    - points (B, N, F), transf_matrix (F+1, F+1)
    all sequences of points are transformed using the same matrix and the output points have shape (B, N, F).
    transf_matrix is broadcasted.
    - points (B, N, F), transf_matrix (B, F+1, F+1)
    each sequence of points is transformed using its own matrix and the output points have shape (B, N, F).
    Note this function assumes points.shape[-1] == matrix.shape[-1] - 1, which means that last
    rows in the matrices do not influence the final results.
    For 2D points only the first 2x3 parts of the matrices will be used.

    :param points: Input points of shape (N, F) or (B, N, F)
        with F = 2 or 3 depending on input points are 2D or 3D points.
    :param transf_matrix: Transformation matrix of shape (F+1, F+1) or (B, F+1, F+1) with F = 2 or 3.
    :return: Transformed points of shape (N, F) or (B, N, F) depending on the dimensions of the input points.
    """
    points_log = f" received points with shape {points.shape} "
    matrix_log = f" received matrices with shape {transf_matrix.shape} "

    assert points.ndim in [2, 3], f"points should have ndim in [2,3],{points_log}"
    assert transf_matrix.ndim in [2, 3], f"matrix should have ndim in [2,3],{matrix_log}"
    assert points.ndim >= transf_matrix.ndim, f"points ndim should be >= than matrix,{points_log},{matrix_log}"

    points_feat = points.shape[-1]
    assert points_feat in [2, 3], f"last points dimension must be 2 or 3,{points_log}"
    assert transf_matrix.shape[-1] == transf_matrix.shape[-2], f"matrix should be a square matrix,{matrix_log}"

    matrix_feat = transf_matrix.shape[-1]
    assert matrix_feat in [3, 4], f"last matrix dimension must be 3 or 4,{matrix_log}"
    assert points_feat == matrix_feat - 1, f"points last dim should be one less than matrix,{points_log},{matrix_log}"

    def _transform(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
        num_dims = transf_matrix.shape[-1] - 1
        transf_matrix = np.transpose(transf_matrix, (0, 2, 1))
        return points @ transf_matrix[:, :num_dims, :num_dims] + transf_matrix[:, -1:, :num_dims]

    if points.ndim == transf_matrix.ndim == 2:
        points = np.expand_dims(points, 0)
        transf_matrix = np.expand_dims(transf_matrix, 0)
        return _transform(points, transf_matrix)[0]

    elif points.ndim == transf_matrix.ndim == 3:
        return _transform(points, transf_matrix)

    elif points.ndim == 3 and transf_matrix.ndim == 2:
        transf_matrix = np.expand_dims(transf_matrix, 0)
        return _transform(points, transf_matrix)
    else:
        raise NotImplementedError(f"unsupported case!{points_log},{matrix_log}")


def transform_point(point: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """Transform a single vector using transformation matrix.
    This function call transform_points internally

    :param point: vector of shape (N)
    :param transf_matrix: transformation matrix of shape (N+1, N+1)
    :return: vector of same shape as input point
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
