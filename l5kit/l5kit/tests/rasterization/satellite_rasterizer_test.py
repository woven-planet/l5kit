import cv2
import numpy as np

from l5kit.data.zarr_dataset import AGENT_DTYPE, FRAME_DTYPE
from l5kit.geometry import yaw_as_rotation33
from l5kit.rasterization import SatelliteRasterizer


def test_satellite_rasterizer() -> None:

    sat_image = np.zeros((10001, 10001, 3), dtype=np.uint8)
    center_point = np.array([5000, 5000, 0])
    sat_image[center_point[0] - 50 : center_point[0] + 50, center_point[1] - 50 : center_point[1] + 50] = 255

    map_to_sat = np.eye(4)
    rast = SatelliteRasterizer(
        raster_size=(500, 500),
        pixel_size=np.array([2.0, 2.0]),
        ego_center=np.array([0.5, 0.5]),
        map_im=sat_image,
        map_to_sat=map_to_sat,
        interpolation=cv2.INTER_NEAREST,
    )

    # TODO remove hack: update the values in this test, this doesn't correspond to the map_to_sat transform scaling
    rast.map_pixel_scale = 5.0

    for yaw in [0, 0.321 * np.pi, 0.5 * np.pi, 1.0 * np.pi]:
        frames = np.zeros(1, dtype=FRAME_DTYPE)
        frames[0]["ego_translation"] = center_point
        frames[0]["ego_rotation"] = yaw_as_rotation33(yaw)
        crop_im = rast.rasterize(frames, np.zeros(0, AGENT_DTYPE))

        box_area_expected = (100 * (5 / 2)) ** 2
        assert np.abs(crop_im.mean(axis=2).sum() - box_area_expected) < 5  # Within 5 pixels error
