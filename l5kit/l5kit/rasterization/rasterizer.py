from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


# Ford Fusion dimensions TODO move somewhere else
EGO_EXTENT_WIDTH = 1.85
EGO_EXTENT_LENGTH = 4.87
EGO_EXTENT_HEIGHT = 1.8  # Height includes sensor


class Rasterizer(ABC):
    """Base class for something that takes a single state of the world, and outputs a (multi-channel) image.
    """

    def __init__(self) -> None:
        pass  # TODO are we sure we don't want at least the pixel information here?

    @abstractmethod
    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        pass

    @abstractmethod
    def num_channels(self) -> int:
        pass
