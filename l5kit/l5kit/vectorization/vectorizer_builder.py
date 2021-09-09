import numpy as np

from l5kit.vectorization.vectorizer import Vectorizer
from l5kit.rasterization.rasterizer_builder import get_hardcoded_world_to_ecef
from l5kit.data.map_api import CACHE_SIZE, InterpolationMethod, MapAPI
from ..data import DataManager
from l5kit.configs.config import load_metadata

def build_vectorizer(cfg: dict, data_manager: DataManager) -> Vectorizer:
    """Factory function for rasterizers, reads the config, loads required data and initializes the correct rasterizer.

    Args:
        cfg (dict): Config.
        data_manager (DataManager): Datamanager that is used to require files to be present.

    Raises:
        NotImplementedError: Thrown when the ``map_type`` read from the config doesn't have an associated rasterizer
        type in this factory function. If you have custom rasterizers, you can wrap this function in your own factory
        function and catch this error.

    Returns:
        Rasterizer: Rasterizer initialized given the supplied config.
    """
    dataset_meta_key = cfg["raster_params"]["dataset_meta_key"] # TODO positioning of key
    try:
        dataset_meta = load_metadata(data_manager.require(dataset_meta_key))
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
    except (KeyError, FileNotFoundError):  # TODO remove when new dataset version is available
        world_to_ecef = get_hardcoded_world_to_ecef()

    mapAPI = MapAPI(data_manager.require(cfg["raster_params"]["semantic_map_key"]), world_to_ecef)
    vectorizer = Vectorizer(cfg, mapAPI)

    return vectorizer