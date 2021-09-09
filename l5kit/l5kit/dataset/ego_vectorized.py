
from typing import Optional
from functools import partial

import numpy as np

from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer
from l5kit.dataset import EgoDataset
from l5kit.data import ChunkedDataset
from l5kit.rasterization import RenderContext
from l5kit.vectorization.vectorizer import Vectorizer
from ..sampling import generate_agent_sample_vectorized

class EgoDatasetVectorized(EgoDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        vectorizer: Vectorizer,
        perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNNs with vectorized input

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            rasterizer (Rasterizer): an object that support rasterisation around an agent (AV or not) - optional for viz [TODO]
            vectorizer (Vectorizer): a object that supports vectorization around an AV
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
        None if not desired 
        """
        super().__init__(cfg, zarr_dataset, rasterizer, perturbation)

        # replace the sample function to access other agents
        # TODO: lberg to check - comment not understandable
        render_context = RenderContext(
            raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
            set_origin_to_bottom=cfg["raster_params"]["set_origin_to_bottom"],
        )
        self.sample_function = partial(
            generate_agent_sample_vectorized,
            render_context=render_context,
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            step_time=cfg["model_params"]["step_time"],
            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=rasterizer,
            perturbation=perturbation,
            vectorizer=vectorizer
        )

    def get_scene_dataset(self, scene_index: int) -> "EgoDatasetVectorized":
        dataset = super().get_scene_dataset(scene_index).dataset
        return EgoDatasetVectorized(self.cfg, dataset, self.rasterizer, self.perturbation)