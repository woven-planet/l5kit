from typing import Iterable, Tuple

import cv2
import imageio
import numpy as np


def write_gif(
        output_filepath: str,
        images: Iterable[np.ndarray],
        resolution: Tuple[int, int],
        fps: float = 24.0,
        loop: int = 0,
        interpolation: int = cv2.INTER_CUBIC,
) -> None:
    """Writes input RGB images to given output gif filepath using imageio. It resizes images
    if necessary using given interpolation (default=``cv2.INTER_CUBIC``).

    Arguments:
        output_filepath (str): output filepath, should end in .gif
        images (Iterable[np.ndarray]): a list or other iterable of images.
        resolution (Tuple[int, int]): desired resolution, e.g. (512, 512)

    Keyword Arguments:
        fps (float): Frames per second (default: {24.0})
        loop (int): 0 means loop forever, any other number loops the GIF that many times (default: {0})
        interpolation (int): Interpolation to be used when resizing (default: {cv2.INTER_CUBIC})
    """

    duration = 1 / fps
    resized_images = []

    for img in images:
        # Go from C,0,1 to 0,1,C ordering
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        if img.shape[:2] != resolution:
            img = cv2.resize(img, resolution, interpolation=interpolation)
        resized_images.append(img)

    imageio.mimwrite(output_filepath, resized_images, duration=duration, loop=loop)
