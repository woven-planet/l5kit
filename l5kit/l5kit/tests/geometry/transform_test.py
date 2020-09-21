import numpy as np
import pytest
import transforms3d

from l5kit.geometry import transform_point, transform_points, transform_points_transposed
from l5kit.rasterization.render_context import RenderContext


def test_transform_to_image_space_2d() -> None:

    image_shape = (200, 200)
    pixel_size = np.asarray((1.0, 0.5))
    offset = np.asarray((0, -2))

    render_context = RenderContext(
        raster_size_px=image_shape,
        pixel_size_m=pixel_size,
        center_in_raster_ratio=offset,
    )

    input_points = np.array([[0, 0], [10, 10], [-10, -10]])
    expected_output_points = np.array([[100, 104], [110, 124], [90, 84]])

    tf = render_context.raster_from_local
    output_points = transform_points(input_points, tf)

    np.testing.assert_array_equal(output_points, expected_output_points)


def test_transform_single_point() -> None:

    shape = (200, 200)
    pixel_size = np.asarray((1.0, 0.5))
    offset = np.asarray((0, -2))

    point = np.array([10, 10])
    expected_point = np.array([110, 124])

    tf = raster_from_world(shape, pixel_size, offset)
    output_point = transform_point(point, tf)

    np.testing.assert_array_equal(output_point, expected_point)


def test_transform_points_transpose_equivalence() -> None:
    input_points = np.random.rand(10, 3)
    input_points_t = input_points.transpose(1, 0)

    #  Generate some random transformation matrix
    tf = np.identity(4)
    tf[:3, :3] = transforms3d.euler.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
    tf[3, :3] = np.random.rand(3)

    output_points = transform_points(input_points, tf)
    output_points_t = transform_points_transposed(input_points_t, tf)

    np.testing.assert_array_equal(output_points.transpose(1, 0), output_points_t)
    tf_inv = np.linalg.inv(tf)

    input_points_recovered = transform_points(output_points, tf_inv)
    input_points_t_recovered = transform_points_transposed(output_points_t, tf_inv)

    np.testing.assert_almost_equal(input_points_recovered, input_points, decimal=10)
    np.testing.assert_almost_equal(input_points_t_recovered, input_points_t, decimal=10)


def test_wrong_input_shape() -> None:
    tf = np.eye(4)

    with pytest.raises(ValueError):
        points = np.zeros((3, 10))
        transform_points(points, tf)

    with pytest.raises(ValueError):
        points = np.zeros((10, 3))
        transform_points_transposed(points, tf)
