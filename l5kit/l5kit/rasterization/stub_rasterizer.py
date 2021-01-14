from typing import List, Optional

import numpy as np

from .rasterizer import Rasterizer
from .render_context import RenderContext


class StubRasterizer(Rasterizer):
    """This rasterizer doesn't actually do anything, it returns an all-black image. Useful for testing.

    """

    def __init__(self, render_context: RenderContext):
        """

        Args:
            render_context (RenderContext): The render context used to convert coordinates into raster space
            filter_agents_threshold (float): Value between 0 and 1 used to filter uncertain agent detections
        """
        super(StubRasterizer, self).__init__()
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Just return an empty raster_size x 3 black image
        """
        out_im = np.zeros((self.raster_size[0], self.raster_size[1], 3), dtype=np.float32)
        return out_im

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        """
        Return a completely black image.
        """
        return np.zeros((self.raster_size[0], self.raster_size[1], 3), dtype=np.uint8)

    def num_channels(self) -> int:
        return 3
