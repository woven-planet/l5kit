import numpy as np

from l5kit.geometry import transform_points
from l5kit.rasterization.render_context import RenderContext


def test_transform_to_image_space_2d() -> None:
    image_shape = np.asarray((200, 200))
    center_in_raster_ratio = np.asarray((0.5, 0.5))
    pixel_size = np.asarray((1.0, 1.0))
    center_world = np.asarray((0, -2))

    render_context = RenderContext(
        raster_size_px=image_shape, pixel_size_m=pixel_size, center_in_raster_ratio=center_in_raster_ratio
    )

    input_points = np.array([[0, 0], [10, 10], [-10, -10]])
    expected_output_points = np.array([[100, 102], [110, 112], [90, 92]])

    tf = render_context.raster_from_world(center_world, 0.0)
    output_points = transform_points(input_points, tf)

    np.testing.assert_array_equal(output_points, expected_output_points)
