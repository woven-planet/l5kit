from typing import List, Optional, Tuple

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
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        ego_center: np.ndarray,
        filter_agents_threshold: float,
        history_num_frames: int,
        map_im: np.ndarray,
        world_to_aerial: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        super(SatBoxRasterizer, self).__init__()
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames

        self.map_im = map_im
        self.world_to_aerial = world_to_aerial
        self.interpolation = interpolation

        self.box_rast = BoxRasterizer(render_context, raster_size, filter_agents_threshold, history_num_frames)
        self.sat_rast = SatelliteRasterizer(
            render_context, raster_size, pixel_size, map_im, world_to_aerial, interpolation
        )

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
