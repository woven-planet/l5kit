
from functools import partial
from typing import Callable, Optional

from l5kit.data import ChunkedDataset
from l5kit.dataset import BaseEgoDataset
from l5kit.kinematic import Perturbation
from l5kit.vectorization.vectorizer import Vectorizer

from ..sampling import generate_agent_sample_vectorized


class EgoDatasetVectorized(BaseEgoDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        vectorizer: Vectorizer,
        perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNNs with vectorized input

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            vectorizer (Vectorizer): a object that supports vectorization around an AV
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
        None if not desired
        """
        self.perturbation = perturbation
        self.vectorizer = vectorizer
        super().__init__(cfg, zarr_dataset)

    def _get_sample_function(self) -> Callable[..., dict]:
        return partial(
            generate_agent_sample_vectorized,
            history_num_frames_ego=self.cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=self.cfg["model_params"]["history_num_frames_agents"],
            future_num_frames=self.cfg["model_params"]["future_num_frames"],
            step_time=self.cfg["model_params"]["step_time"],
            filter_agents_threshold=self.cfg["raster_params"]["filter_agents_threshold"],
            perturbation=self.perturbation,
            vectorizer=self.vectorizer
        )

    def get_scene_dataset(self, scene_index: int) -> "EgoDatasetVectorized":
        dataset = super().get_scene_dataset(scene_index).dataset
        return EgoDatasetVectorized(self.cfg, dataset, self.vectorizer, self.perturbation)
