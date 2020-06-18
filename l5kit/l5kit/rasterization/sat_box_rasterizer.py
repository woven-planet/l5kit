from typing import Optional, Tuple

import cv2
import numpy as np

from .box_rasterizer import BoxRasterizer
from .rasterizer import Rasterizer
from .satellite_rasterizer import SatelliteRasterizer


class SatBoxRasterizer(Rasterizer):
    """Combine a Satellite and a Box Rasterizers into a single class
    """

    def __init__(
        self,
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        ego_center: np.ndarray,
        filter_agents_threshold: float,
        map_im: np.ndarray,
        map_to_sat: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        super(SatBoxRasterizer, self).__init__()
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center
        self.filter_agents_threshold = filter_agents_threshold
        self.map_im = map_im
        self.map_to_sat = map_to_sat
        self.interpolation = interpolation

        self.box_rast = BoxRasterizer(raster_size, pixel_size, ego_center, filter_agents_threshold)
        self.sat_rast = SatelliteRasterizer(raster_size, pixel_size, ego_center, map_im, map_to_sat, interpolation)

    def rasterize(
        self, history_frames: np.ndarray, all_agents: np.ndarray, agent: Optional[np.ndarray] = None
    ) -> np.ndarray:
        im_out_box = self.box_rast.rasterize(history_frames, all_agents, agent)
        im_out_sat = self.sat_rast.rasterize(history_frames, all_agents, agent)
        return np.concatenate([im_out_box, im_out_sat], -1)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        im_out_box = self.box_rast.to_rgb(in_im[..., :-3], **kwargs)
        im_out_sat = self.sat_rast.to_rgb(in_im[..., -3:], **kwargs)

        # merge the two together using box as mask
        mask = np.any(im_out_box > 0, axis=-1)
        im_out_sat[mask] = im_out_box[mask]
        return im_out_sat
