import cv2
import numpy as np


def crop_rectangle_from_image(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """``crop_rectangle_from_image`` takes an image and 4 corners in pixel coordinates, it returns
    the sub-image inside that cropped area, rotated upright.

    Args:
        image (np.ndarray): image to crop from
        corners (np.ndarray): corners, array of shape (4,2)

    Returns:
        np.ndarray: crop from input containing the corners
    """
    rect = cv2.minAreaRect(np.array(corners[:, ::-1], dtype=np.float32))
    center, size, theta = rect
    center, size = tuple(center), tuple(np.int0(size))

    M = cv2.getRotationMatrix2D(center, theta, 1)
    dest = cv2.warpAffine(image, M, tuple((np.array(image.shape[:2]) * 2).astype(np.int64)))

    out = cv2.getRectSubPix(dest, size, center)
    return out
