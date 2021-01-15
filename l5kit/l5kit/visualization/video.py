from typing import Iterable, Tuple

import cv2
import numpy as np


def write_video(
        output_filepath: str,
        images: Iterable[np.ndarray],
        resolution: Tuple[int, int],
        fps: float = 24.0,
        codec: str = "FMP4",
        interpolation: int = cv2.INTER_CUBIC,
) -> None:
    """Writes input RGB images to given output video filepath using OpenCV. It resizes images if
    necessary using given interpolation (default=``cv2.INTER_CUBIC``).

    Note that as this function uses OpenCV, your image's color channels will be inverted (RGB -> BGR) prior to writing.

    Arguments:
        output_filepath (str): output filepath, generally this should end in .mp4 or .avi
        depending on the codec used.
        images (Iterable[np.ndarray]): a list or other iterable of images.
        resolution (Tuple[int, int]): video resolution, e.g. (512, 512), the input frames are resized to this.

    Keyword Arguments:
        fps (float): Frames per second (default: {24.0})
        codec (str): Codec to be used. Note that with X264 codec only certain resolutions may work (default: {"FMP4"})
        interpolation (int): Interpolation to be used when resizing (default: {cv2.INTER_CUBIC})
    """

    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(output_filepath, fourcc, fps, resolution)

    for img in images:
        # Go from C,0,1 to 0,1,C ordering
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        if img.shape[:2] != resolution:
            img = cv2.resize(img, resolution, interpolation=interpolation)
        img = img[..., ::-1]  # RGB -> BGR
        vw.write(img)

    vw.release()
