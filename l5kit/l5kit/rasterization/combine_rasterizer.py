from typing import List, Optional, Callable

import numpy as np

from .rasterizer import Rasterizer


class CombineRasterizer(Rasterizer):
    """This rasterizer combines multiple rasterizers' output.
    """

    def __init__(self, rasterizers: List[Rasterizer], to_rgb_fn: Callable):
        """

        Args:
            rasterizers (List[Rasterizer]): Rasterizers to combine
        """
        super(CombineRasterizer, self).__init__()
        self.rasterizers = rasterizers
        self.to_rgb_fn = to_rgb_fn

    def rasterize(
        self, history_frames: np.ndarray, all_agents: np.ndarray, agent: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Rasterize the history wrt to the first element in history_frames (most recent).
        This concatenates the rasters on the colour channel dimension.
        """

        rasters = [r.rasterize(history_frames, all_agents, agent) for r in self.rasterizers]
        return np.concatenate(rasters, axis=2)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return self.to_rgb_fn(in_im, **kwargs)
