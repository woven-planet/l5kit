import cv2
import numpy as np
import pytest

from l5kit.rasterization import get_sat_image_crop, get_sat_image_crop_scaled


def test_satellite_image_cropping_square() -> None:
    sat_image = np.zeros((1001, 1001, 3), dtype=np.float32)
    sat_image[500:, :500, 0] = 1  # Make top right corner red
    sat_image[:500, :500, 1] = 1  # Make bottom right corner green
    sat_image[:500, 500:, 2] = 1  # Make bottom left corner blue

    for yaw in [None, 0.5 * np.pi, 0.25 * np.pi, 0.1234 * np.pi]:
        crop_im = get_sat_image_crop(
            sat_image, crop_size=(200, 200), sat_pixel_translation=np.array([500, 500]), yaw=yaw
        )
        assert crop_im.shape == (200, 200, 3)

        assert crop_im[..., 0].mean() == pytest.approx(0.25, 0.1)  # One quarter should still be red
        assert crop_im[..., 1].mean() == pytest.approx(0.25, 0.1)  # One quarter should still be green
        if yaw == 0.5 * np.pi:
            # The top left should be blue now
            np.testing.assert_array_equal(crop_im[0, 0], np.array([0, 0, 1]))

    with pytest.raises(IndexError):
        get_sat_image_crop_scaled(sat_image, crop_size=(10, 10), sat_pixel_translation=np.array([0, 1000]))


def test_satellite_image_cropping_rectangular() -> None:
    sat_image = np.zeros((1001, 1001, 3), dtype=np.float32)
    sat_image[500:, :500, 0] = 1  # Make top right corner red
    sat_image[:500, :500, 1] = 1  # Make bottom right corner green
    sat_image[:500, 500:, 2] = 1  # Make bottom left corner blue

    for yaw in [None, np.pi, 0.5 * np.pi]:
        crop_im = get_sat_image_crop_scaled(
            sat_image, crop_size=(200, 100), sat_pixel_translation=np.array([500, 500]), yaw=yaw
        )
        assert crop_im.shape == (100, 200, 3)

        assert crop_im[..., 0].mean() == pytest.approx(0.25, 0.1)  # One quarter should still be red
        assert crop_im[..., 1].mean() == pytest.approx(0.25, 0.1)  # One quarter should still be green
        if yaw == 0.5 * np.pi:
            # The top left should be blue now
            np.testing.assert_array_equal(crop_im[0, 0], np.array([0, 0, 1]))

    with pytest.raises(IndexError):
        get_sat_image_crop_scaled(sat_image, crop_size=(10, 10), sat_pixel_translation=np.array([0, 1000]))


def test_satellite_image_cropping_scaled() -> None:
    sat_image = np.zeros((10001, 10001, 3), dtype=np.float32)
    center_point = np.array([5000, 5000])
    sat_image[center_point[0] - 50: center_point[0] + 50, center_point[1] - 50: center_point[1] + 50] = 1.0

    for yaw in [None, 0.321 * np.pi, 0.5 * np.pi, 1.0 * np.pi]:
        crop_im = get_sat_image_crop_scaled(
            sat_image,
            crop_size=(500, 500),
            sat_pixel_translation=center_point,
            yaw=yaw,
            sat_pixel_scale=5.0,  # One pixel in satellite image is 5 meters in the real world
            pixel_size=2.0,  # We want our output image to have pixels corresponding to 2 meters in the real world
            interpolation=cv2.INTER_NEAREST,
        )
        box_area_expected = (100 * (5 / 2)) ** 2
        assert np.abs(crop_im.mean(axis=2).sum() - box_area_expected) < 5  # Within 5 pixels error
