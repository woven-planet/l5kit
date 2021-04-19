from typing import List, Optional

import cv2
import numpy as np

from .box_rasterizer import BoxRasterizer
from .rasterizer import Rasterizer
from .render_context import RenderContext
from .satellite_rasterizer import SatelliteRasterizer


class SatBoxRasterizer(Rasterizer):
    """Combine a Satellite and a Box Rasterizers into a single class
    """

    def __init__(
            self,
            render_context: RenderContext,
            filter_agents_threshold: float,
            history_num_frames: int,
            map_im: np.ndarray,
            world_to_aerial: np.ndarray,
            interpolation: int = cv2.INTER_LINEAR,
            render_ego_history: bool = True,
    ):
        super(SatBoxRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames

        self.map_im = map_im
        self.world_to_aerial = world_to_aerial
        self.interpolation = interpolation

        self.box_rast = BoxRasterizer(render_context, filter_agents_threshold, history_num_frames, render_ego_history)
        self.sat_rast = SatelliteRasterizer(render_context, map_im, world_to_aerial, interpolation)

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        im_out_box = self.box_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
        im_out_sat = self.sat_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
        return np.concatenate([im_out_box, im_out_sat], -1)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        im_out_box = self.box_rast.to_rgb(in_im[..., :-3], **kwargs)
        im_out_sat = self.sat_rast.to_rgb(in_im[..., -3:], **kwargs)

        # merge the two together using box as mask
        mask = np.any(im_out_box > 0, axis=-1)
        im_out_sat[mask] = im_out_box[mask]
        return im_out_sat

    def num_channels(self) -> int:
        return self.box_rast.num_channels() + self.sat_rast.num_channels()
