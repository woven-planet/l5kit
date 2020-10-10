from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .rasterizer import Rasterizer
from .render_context import RenderContext


class MapAndBoxRasterizer(Rasterizer, ABC):
    """A rasterizer that combines a Map and a Box Rasterizers into a single class"""

    def __init__(
        self, render_context: RenderContext, filter_agents_threshold: float, history_num_frames: int,
    ) -> None:
        super().__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames

    @property
    @abstractmethod
    def box_rast(self) -> Rasterizer:
        pass

    @property
    @abstractmethod
    def map_rast(self) -> Rasterizer:
        pass

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        im_out_box = self.box_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
        im_out_map = self.map_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
        return np.concatenate([im_out_box, im_out_map], -1)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        im_out_box = self.box_rast.to_rgb(in_im[..., :-3], **kwargs)
        im_out_map = self.map_rast.to_rgb(in_im[..., -3:], **kwargs)
        # merge the two together
        mask_box = np.any(im_out_box > 0, -1)
        im_out_map[mask_box] = im_out_box[mask_box]
        return im_out_map
