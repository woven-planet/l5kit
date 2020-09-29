import numpy as np
import pytest
import transforms3d

from l5kit.geometry import transform_point, transform_points, world_to_image_pixels_matrix


def test_transform_to_image_space_2d() -> None:

    image_shape = (200, 200)
    pixel_size = np.asarray((1.0, 0.5))
    offset = np.asarray((0, -2))

    input_points = np.array([[0, 0], [10, 10], [-10, -10]])
    expected_output_points = np.array([[100, 104], [110, 124], [90, 84]])

    tf = world_to_image_pixels_matrix(image_shape, pixel_size, offset)
    output_points = transform_points(input_points, tf)

    np.testing.assert_array_equal(output_points, expected_output_points)


def test_transform_single_point() -> None:

    shape = (200, 200)
    pixel_size = np.asarray((1.0, 0.5))
    offset = np.asarray((0, -2))

    point = np.array([10, 10])
    expected_point = np.array([110, 124])

    tf = world_to_image_pixels_matrix(shape, pixel_size, offset)
    output_point = transform_point(point, tf)

    np.testing.assert_array_equal(output_point, expected_point)


def test_transform_points_revert_equivalence() -> None:
    input_points = np.random.rand(10, 3)

    #  Generate some random transformation matrix
    tf = np.identity(4)
    tf[:3, :3] = transforms3d.euler.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
    tf[3, :3] = np.random.rand(3)

    output_points = transform_points(input_points, tf)

    tf_inv = np.linalg.inv(tf)

    input_points_recovered = transform_points(output_points, tf_inv)

    np.testing.assert_almost_equal(input_points_recovered, input_points, decimal=10)


def test_wrong_input_shape() -> None:
    tf = np.eye(4)

    with pytest.raises(AssertionError):
        points = np.zeros((3, 10))
        transform_points(points, tf)

    with pytest.raises(AssertionError):
        points = np.zeros((10, 4))  # should be 3D for a 4D matrix
        transform_points(points, tf)

    with pytest.raises(AssertionError):
        points = np.zeros((10, 3))  # should be 2D for a 3D matrix
        transform_points(points, tf[:3, :3])
