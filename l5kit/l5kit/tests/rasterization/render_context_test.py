import numpy as np
import pytest

from l5kit.geometry import transform_points
from l5kit.rasterization.render_context import RenderContext


@pytest.mark.parametrize("set_origin_to_bottom", [False, True])
def test_transform_points_to_raster(set_origin_to_bottom: bool) -> None:
    image_shape_px = np.asarray((200, 200))
    center_in_raster_ratio = np.asarray((0.5, 0.5))
    pixel_size_m = np.asarray((1.0, 1.0))
    center_world = np.asarray((0, -2))

    render_context = RenderContext(
        raster_size_px=image_shape_px,
        pixel_size_m=pixel_size_m,
        center_in_raster_ratio=center_in_raster_ratio,
        set_origin_to_bottom=set_origin_to_bottom,
    )

    input_points = np.array([[0, 0], [10, 10], [-10, -10]])
    if set_origin_to_bottom:
        expected_output_points = np.array([[100, 98], [110, 88], [90, 108]])
    else:
        expected_output_points = np.array([[100, 102], [110, 112], [90, 92]])

    tf = render_context.raster_from_world(center_world, 0.0)
    output_points = transform_points(input_points, tf)

    np.testing.assert_array_equal(output_points, expected_output_points)
