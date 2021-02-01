from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Perturbation(ABC):
    @abstractmethod
    def perturb(self, history_frames: np.ndarray, future_frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            history_frames (np.ndarray): array of past frames
            future_frames (np.ndarray): array of future frames
            kwargs: optional extra arguments for the specific perturber

        Returns:
            history_frames (np.ndarray): array of past frames with perturbation applied
            future_frames (np.ndarray): array of future frames with perturbation applied
        """
        pass
