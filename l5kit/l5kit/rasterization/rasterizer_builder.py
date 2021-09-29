import cv2
import numpy as np

from l5kit.configs.config import load_metadata

from ..data import DataManager
from .box_rasterizer import BoxRasterizer
from .rasterizer import Rasterizer
from .render_context import RenderContext
from .sat_box_rasterizer import SatBoxRasterizer
from .satellite_rasterizer import SatelliteRasterizer
from .sem_box_rasterizer import SemBoxRasterizer
from .semantic_rasterizer import SemanticRasterizer
from .stub_rasterizer import StubRasterizer


def _load_satellite_map(image_key: str, data_manager: DataManager) -> np.ndarray:
    """Loads image from given key.

    Args:
        image_key (str): key to the image (e.g. ``maps/my_satellite_image.png``)
        data_manager (DataManager): DataManager used for requiring files

    Returns:
        np.ndarry: Image
    """

    image_path = data_manager.require(image_key)
    image = cv2.imread(image_path)[..., ::-1]  # BGR->RGB
    if image is None:
        raise Exception(f"Failed to load image from {image_path}")

    return image


def build_rasterizer(cfg: dict, data_manager: DataManager) -> Rasterizer:
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
    raster_cfg = cfg["raster_params"]
    map_type = raster_cfg["map_type"]
    dataset_meta_key = raster_cfg["dataset_meta_key"]

    render_context = RenderContext(
        raster_size_px=np.array(raster_cfg["raster_size"]),
        pixel_size_m=np.array(raster_cfg["pixel_size"]),
        center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
        set_origin_to_bottom=raster_cfg["set_origin_to_bottom"],
    )

    filter_agents_threshold = raster_cfg["filter_agents_threshold"]
    history_num_frames = cfg["model_params"]["history_num_frames"]
    render_ego_history = cfg["model_params"]["render_ego_history"]

    if map_type in ["py_satellite", "satellite_debug"]:
        sat_image = _load_satellite_map(raster_cfg["satellite_map_key"], data_manager)

        dataset_meta = load_metadata(data_manager.require(dataset_meta_key))
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        ecef_to_aerial = np.array(dataset_meta["ecef_to_aerial"], dtype=np.float64)

        world_to_aerial = np.matmul(ecef_to_aerial, world_to_ecef)
        if map_type == "py_satellite":
            return SatBoxRasterizer(
                render_context, filter_agents_threshold, history_num_frames, sat_image, world_to_aerial,
                render_ego_history=render_ego_history
            )
        else:
            return SatelliteRasterizer(render_context, sat_image, world_to_aerial)

    elif map_type in ["py_semantic", "semantic_debug"]:
        semantic_map_filepath = data_manager.require(raster_cfg["semantic_map_key"])
        dataset_meta = load_metadata(data_manager.require(dataset_meta_key))
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

        if map_type == "py_semantic":
            return SemBoxRasterizer(
                render_context, filter_agents_threshold, history_num_frames, semantic_map_filepath, world_to_ecef,
                render_ego_history=render_ego_history
            )
        else:
            return SemanticRasterizer(render_context, semantic_map_filepath, world_to_ecef)

    elif map_type == "box_debug":
        return BoxRasterizer(render_context, filter_agents_threshold, history_num_frames,
                             render_ego_history=render_ego_history)
    elif map_type == "stub_debug":
        return StubRasterizer(render_context)
    else:
        raise NotImplementedError(f"Rasterizer for map type {map_type} is not supported.")
