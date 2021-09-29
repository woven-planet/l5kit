import numpy as np

from l5kit.configs.config import load_metadata
from l5kit.data.map_api import MapAPI
from l5kit.vectorization.vectorizer import Vectorizer

from ..data import DataManager


def build_vectorizer(cfg: dict, data_manager: DataManager) -> Vectorizer:
    """Factory function for vectorizers, reads the config, loads required data and initializes the vectorizer.

    Args:
        cfg (dict): Config.
        data_manager (DataManager): Datamanager that is used to require files to be present.

    Returns:
        Vectorizer: Vectorizer initialized given the supplied config.
    """
    dataset_meta_key = cfg["raster_params"]["dataset_meta_key"]  # TODO positioning of key
    dataset_meta = load_metadata(data_manager.require(dataset_meta_key))
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    mapAPI = MapAPI(data_manager.require(cfg["raster_params"]["semantic_map_key"]), world_to_ecef)
    vectorizer = Vectorizer(cfg, mapAPI)

    return vectorizer
