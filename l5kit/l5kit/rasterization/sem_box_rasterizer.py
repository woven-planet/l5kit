from typing import Optional, Tuple

import numpy as np

from .box_rasterizer import BoxRasterizer
from .rasterizer import Rasterizer
from .semantic_rasterizer import SemanticRasterizer


class SemBoxRasterizer(Rasterizer):
    """Combine a Semantic Map and a Box Rasterizers into a single class
    """

    def __init__(
        self,
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        ego_center: np.ndarray,
        filter_agents_threshold: float,
        history_num_frames: int,
        semantic_map: dict,
        pose_to_ecef: np.ndarray,
    ):
        super(SemBoxRasterizer, self).__init__()
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames

        self.box_rast = BoxRasterizer(raster_size, pixel_size, ego_center, filter_agents_threshold, history_num_frames)
        self.sat_rast = SemanticRasterizer(raster_size, pixel_size, ego_center, semantic_map, pose_to_ecef)

    def rasterize(
        self, history_frames: np.ndarray, all_agents: np.ndarray, agent: Optional[np.ndarray] = None
    ) -> np.ndarray:
        im_out_box = self.box_rast.rasterize(history_frames, all_agents, agent)
        im_out_sat = self.sat_rast.rasterize(history_frames, all_agents, agent)
        return np.concatenate([im_out_box, im_out_sat], -1)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        im_out_box = self.box_rast.to_rgb(in_im[..., :-3], **kwargs)
        im_out_sat = self.sat_rast.to_rgb(in_im[..., -3:], **kwargs)
        # merge the two together
        mask_box = np.any(im_out_box > 0, -1)
        im_out_sat[mask_box] = im_out_box[mask_box]
        return im_out_sat
