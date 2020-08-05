import numpy as np
import pytest

from l5kit.geometry import voxel


def test_bounds_check() -> None:
    grid_shape = (10, 10, 10)
    pixel_points = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 0], [1, 0, 0], [10, 0, 0]])

    within_bounds = voxel.points_within_bounds(pixel_points, grid_shape)
    np.testing.assert_array_equal(within_bounds, [True, True, False, True, False])


def test_intensity_grid() -> None:
    grid_shape = (2, 2, 2)
    pixel_points = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0]])

    expected_intensity_grid = np.zeros(grid_shape)
    expected_intensity_grid[0, 0, 0] = 2
    expected_intensity_grid[1, 0, 0] = 1

    intensity_grid = voxel.voxel_coords_to_intensity_grid(pixel_points, grid_shape)

    np.testing.assert_array_equal(intensity_grid, expected_intensity_grid)

    with pytest.raises(IndexError):
        intensity_grid = voxel.voxel_coords_to_intensity_grid(pixel_points, grid_shape, drop_out_of_bounds=False)


def test_normalize_intensity() -> None:
    intensity_grid = np.array([0, 1, 2, 3])
    max_intensity = 2
    intensity_grid_normalized_expected = np.array([0, 0.5, 1.0, 1.0])

    intensity_grid_normalized = voxel.normalize_intensity(intensity_grid, max_intensity)

    np.testing.assert_array_equal(intensity_grid_normalized, intensity_grid_normalized_expected)
