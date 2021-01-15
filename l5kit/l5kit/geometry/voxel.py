from typing import Collection, Union

import numpy as np


def points_within_bounds(coords: np.ndarray, shape: Union[Collection[int], np.ndarray]) -> np.ndarray:
    """
    Arguments:
        coords (np.ndarray): (N,3)-shaped array containing points.
        shape (tuple of ints or np.ndarray): shape to use as bounds, should be length 3
    Returns:
        Binary mask for given coords array which is True for points that fall within the bounds of shape.
    """
    shape = np.array(shape)
    within_bounds = np.all(coords >= 0, axis=1) * np.all(coords < shape, axis=1)
    return within_bounds


def voxel_coords_to_intensity_grid(
        voxel_coords: np.ndarray, shape: tuple, dtype: np.dtype = np.float32, drop_out_of_bounds: bool = True
) -> np.ndarray:
    """ Puts coords into a grid: for each grid cell the number of points is written there.

    Arguments:
        voxel_coords (np.ndarray): input array with coords (N,3) in intensity grid
        shape (tuple of ints): intensity grid shape

    Keyword Arguments:
        dtype (data-type): data type for the intensity grid (default: {np.float32})
        drop_out_of_bounds (bool): [description] (default: {True})

    Returns:
        np.ndarray -- Array with given shape, the value of each cell is the amount of coords for that point.
    """
    if drop_out_of_bounds:
        voxel_coords = voxel_coords[points_within_bounds(voxel_coords, shape)]

    grid = np.zeros(shape, dtype=dtype)

    c, intensity = np.unique(voxel_coords, axis=0, return_counts=True)
    grid[c[:, 0], c[:, 1], c[:, 2]] = intensity

    return grid


def normalize_intensity(x: np.ndarray, max_intensity: float) -> np.ndarray:
    """Normalize (divide by max) and clip intensity values to fall between 0 and 1.

    Arguments:
        x (np.npdarray): numpy array of any shape
        max_intensity (float): Maximum intensity value (anything above this will become 1)

    Returns:
        np.ndarray -- array of same type and shape as x with values between 0 and 1 only
    """
    return (x / max_intensity).clip(0, 1)
