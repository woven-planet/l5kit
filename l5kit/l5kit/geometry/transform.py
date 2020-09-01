from typing import Optional, Sequence, Tuple, Union, cast

import numpy as np
import pymap3d as pm
import transforms3d


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


def world_to_image_pixels_matrix(
    image_shape: Tuple[int, int],
    pixel_size_m: np.ndarray,
    ego_translation_m: np.ndarray,
    ego_yaw_rad: Optional[float] = None,
    ego_center_in_image_ratio: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Constructs a transformation matrix from world coordinates to image pixels.
    Note: we're ignoring Z coord, with the assumption that it won't change dramatically.
    This means last row of the matrix will be [0,0,1] and we will transform 2D points in fact (X,Y)

        Args:
        image_shape (Tuple[int, int]): the size of the image
        pixel_size_m (np.ndarray): how many meters a pixel cover in the two directions
        ego_translation_m (np.ndarray): translation of the ego in meters in world-coordinates
        ego_yaw_rad (Optional[float]):if defined, rotation is applied so that ego will always face to the right
         in the resulting image coordinates
        ego_center_in_image_ratio (Optional[np.ndarray]): enables to position the ego in different places
         in the resulting image. The [0.5, 0.5] value puts it in the center

    Returns:
        np.ndarray: 3x3 transformation matrix
    """

    # Translate world to ego by applying the negative ego translation.
    world_to_ego_in_2d = np.eye(3, dtype=np.float32)
    world_to_ego_in_2d[0:2, 2] = -ego_translation_m[0:2]

    if ego_yaw_rad is not None:
        # Rotate counter-clockwise by negative yaw to align world such that ego faces right.
        world_to_ego_in_2d = yaw_as_rotation33(-ego_yaw_rad) @ world_to_ego_in_2d

    # Scale the meters to pixels.
    world_to_image_scale = np.eye(3)
    world_to_image_scale[0, 0] = 1.0 / pixel_size_m[0]
    world_to_image_scale[1, 1] = 1.0 / pixel_size_m[1]

    # Move so that it is aligned to the defined image center.
    if ego_center_in_image_ratio is None:
        ego_center_in_image_ratio = np.array([0.5, 0.5])
    ego_center_in_pixels = ego_center_in_image_ratio * image_shape
    image_to_ego_center = np.eye(3)
    image_to_ego_center[0:2, 2] = ego_center_in_pixels

    # Construct the whole transform and return it.
    return image_to_ego_center @ world_to_image_scale @ world_to_ego_in_2d


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


def transform_points(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points using transformation matrix.

    Args:
        points (np.ndarray): Input points (Nx2), (Nx3) or (Nx4).
        transf_matrix (np.ndarray): 3x3 or 4x4 transformation matrix for 2D and 3D input respectively

    Returns:
        np.ndarray: array of shape (N,2) for 2D input points, or (N,3) points for 3D input points
    """
    # TODO: Surely we can do this without transposing.
    return transform_points_transposed(points.transpose(1, 0), transf_matrix).transpose(1, 0)


def transform_points_transposed(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points using transformation matrix.

    Args:
        points (np.ndarray): Input points (2xN), (3xN) or (4xN).
        transf_matrix (np.ndarray): 3x3 or 4x4 transformation matrix for 2D and 3D input respectively

    Returns:
        np.ndarray: array of shape (2,N) for 2D input points, or (3,N) points for 3D input points
    """
    num_dims = transf_matrix.shape[0] - 1
    if points.shape[0] not in [2, 3, 4]:
        raise ValueError("Points input should be (2, N), (3,N) or (4,N) shape, received {}".format(points.shape))

    return transf_matrix.dot(np.vstack((points[:num_dims, :], np.ones(points.shape[1]))))[:num_dims, :]


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


def get_transformation_matrix(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Get a 3D transformation matrix from translation vector and quaternion rotation

    Args:
        translation (np.ndarray): 3D translation vector
        rotation (np.ndarray): 4 quaternion values

    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    rot_mat = transforms3d.quaternions.quat2mat(rotation)
    tr = transforms3d.affines.compose(translation, rot_mat, np.ones(3))
    return tr


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
