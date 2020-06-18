from typing import Optional, Tuple

import numpy as np

from .rasterizer import Rasterizer


class StubRasterizer(Rasterizer):
    """This rasterizer doesn't actually do anything, it returns an all-black image. Useful for testing.

    """

    def __init__(
        self,
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        ego_center: np.ndarray,
        filter_agents_threshold: float,
    ):
        """

        Arguments:
            raster_size {Tuple[int, int]} -- Desired output image size
            pixel_size {np.ndarray} -- Dimensions of one pixel in the real world
            ego_center {np.ndarray} -- Center of ego in the image, [0.5,0.5] would be in the image center.
            filter_agents_threshold {float} -- Value between 0 and 1 used to filter uncertain agent detections
        """
        super(StubRasterizer, self).__init__()
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center
        self.filter_agents_threshold = filter_agents_threshold

    def rasterize(
        self, history_frames: np.ndarray, all_agents: np.ndarray, agent: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Rasterize the history wrt to the first element in history_frames (most recent)
        """
        out_im = np.zeros((len(history_frames) * 2, self.raster_size[0], self.raster_size[1]), dtype=np.float32)
        return out_im

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        """
        Return a completely black image.
        """
        return np.zeros((self.raster_size[0], self.raster_size[1], 3), dtype=np.uint8)
