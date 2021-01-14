from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np

from l5kit.geometry import transform_point


def get_sat_image_crop_scaled_from_ecef(
        sat_image: np.ndarray,
        crop_size: Union[Tuple[int, int], np.ndarray],
        ecef_translation: np.ndarray,
        ecef_to_sat: np.ndarray,
        **kwargs: Any,
) -> np.ndarray:
    """Utility function, calls get_sat_image_crop_scaled, see that function for more details on additional
    keyword arguments (such as ``yaw``).

    Arguments:
        sat_image (np.ndarray): satellite image
        crop_size (Union[Tuple[int, int], np.ndarray]): size of desired crop in pixels
        ecef_translation (np.ndarray): 2D or 3D vector where to center the cropped image
        ecef_to_sat (np.ndarray): transform from ECEF to satellite image space

    Returns:
        np.ndarray -- a crop of satellite_image
    """
    sat_pixel_translation = transform_point(ecef_translation, ecef_to_sat)
    return get_sat_image_crop_scaled(sat_image, crop_size, sat_pixel_translation, **kwargs)


def get_sat_image_crop_scaled(
        sat_image: np.ndarray,
        crop_size: Union[Tuple[int, int], np.ndarray],
        sat_pixel_translation: np.ndarray,
        yaw: Optional[float] = None,
        sat_pixel_scale: float = 1.0,
        pixel_size: float = 1.0,
        interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Calls `get_sat_image_crop` (see that function's docs for further details), and rescales taking
        into account a desired pixel size.

    Example:
        Desired ``crop_size`` is 200x200, and ``pixel_size`` is 0.5: we want an image that corresponds
        to 100x100 meters. This means it extracts a 33x33 image and scales it up to 200x200.

    Arguments:
        sat_image (np.ndarray): satellite image
        crop_size (Union[Tuple[int, int], np.ndarray]): size of desired crop in pixels
        sat_pixel_translation (np.ndarray): 2D or 3D vector where to center the cropped image in pixels.

    Keyword Arguments:
        yaw (Optional[float]): yaw in radians, 0 means no rotation is applied, which generally means up is North.
            default: {None})

        sat_pixel_scale (float): A `sat_pixel_scale` of 3.0 would means that every pixel in the sat
        image corresponds to 3m in the real world. (default: {1.0})
        pixel_size (float): [description] (default: {1.0})
        interpolation (int): [description] (default: {cv2.INTER_LINEAR})

    Returns:
        (np.ndarray): a crop of input ``sat_image``
    """

    max_crop_size = [max(crop_size), max(crop_size)]
    crop_size_in_meters = np.array(max_crop_size) * pixel_size
    crop_size_in_sat_pixels = np.int0(np.round(crop_size_in_meters / sat_pixel_scale))

    sat_crop = get_sat_image_crop(sat_image, crop_size_in_sat_pixels, sat_pixel_translation, yaw)

    resized_sat_crop = cv2.resize(sat_crop, tuple(max_crop_size), interpolation=interpolation)
    start_x = np.int0(resized_sat_crop.shape[0] / 2 - crop_size[1] / 2)
    end_x = start_x + crop_size[1]
    start_y = np.int0(resized_sat_crop.shape[1] / 2 - crop_size[0] / 2)
    end_y = start_y + crop_size[0]

    out_sat_crop = resized_sat_crop[start_x:end_x, start_y:end_y]

    return out_sat_crop


def get_sat_image_crop(
        sat_image: np.ndarray,
        crop_size: Union[Tuple[int, int], np.ndarray],
        sat_pixel_translation: np.ndarray,
        yaw: Optional[float] = None,
) -> np.ndarray:
    """Crops input satellite such that ``sat_pixel_translation`` is centered in the image.

    Arguments:
        sat_image (np.ndarray): satellite image
        crop_size (Union[Tuple[int, int], np.ndarray]): size of desired crop in pixels
        sat_pixel_translation (np.ndarray): 2D or 3D vector where to center the cropped image in pixels.

    Keyword Arguments:
        yaw (Optional[float]): yaw in radians, None or 0 means no rotation is applied to the output image.
            default: {None})

    Returns:
        (np.ndarray): a crop of input ``sat_image``
    """
    if yaw is None:
        return _get_sat_image_crop_without_rotation(sat_image, crop_size, sat_pixel_translation)

    # We scale the image up, so that when it is rotated we can take the center crop without
    # losing the corners
    crop_size_scaled = np.ceil(np.array(crop_size) * np.sqrt(2)).astype(np.int64)

    im = _get_sat_image_crop_without_rotation(sat_image, crop_size_scaled, sat_pixel_translation)

    rot_matrix = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), angle=np.degrees(yaw), scale=1.0)
    rotated_cropped = cv2.warpAffine(im, rot_matrix, (im.shape[1], im.shape[0]))

    # Center crop
    start_x = np.int0(rotated_cropped.shape[0] / 2 - crop_size[0] / 2)
    end_x = start_x + crop_size[0]
    start_y = np.int0(rotated_cropped.shape[1] / 2 - crop_size[1] / 2)
    end_y = start_y + crop_size[1]

    return rotated_cropped[start_x:end_x, start_y:end_y]


def _get_sat_image_crop_without_rotation(
        sat_image: np.ndarray, crop_size: np.ndarray, sat_pixel_translation: np.ndarray
) -> np.ndarray:
    """
    Crops satellite image around given translation.

    sat_pixel_translation should be an array with shape (2,) or (3,).
    """
    assert len(crop_size) >= 2
    assert sat_pixel_translation.shape[0] >= 2

    start_x = np.int0(sat_pixel_translation[0] - crop_size[0] // 2)
    end_x = start_x + crop_size[0]
    start_y = np.int0(sat_pixel_translation[1] - crop_size[1] // 2)
    end_y = start_y + crop_size[1]

    # Crops outside of the image not supported.
    if start_x < 0 or start_y < 0 or end_x >= sat_image.shape[0] or end_y >= sat_image.shape[1]:
        raise IndexError(
            "Satellite image crop ({}-{}, {}-{}) is out of bounds for satellite image of shape {}".format(
                start_x, end_x, start_y, end_y, sat_image.shape
            )
        )

    return sat_image[start_x:end_x, start_y:end_y]
