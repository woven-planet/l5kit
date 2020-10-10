import json
import warnings

import cv2
import numpy as np

from ..data import DataManager
from .box_rasterizer import BoxRasterizer
from .rasterizer import Rasterizer
from .render_context import RenderContext
from .sat_box_rasterizer import SatBoxRasterizer
from .satellite_rasterizer import SatelliteRasterizer
from .sem_box_rasterizer import SemBoxRasterizer
from .semantic_rasterizer import SemanticRasterizer
from .stub_rasterizer import StubRasterizer


def _load_metadata(meta_key: str, data_manager: DataManager) -> dict:
    """
    Load a json metadata file

    Args:
        meta_key (str): relative key to the metadata
        data_manager (DataManager): DataManager used for requiring files

    Returns:
        dict: metadata as a dict
    """
    metadata_path = data_manager.require(meta_key)
    with open(metadata_path, "r") as f:
        metadata: dict = json.load(f)
    return metadata


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


def get_hardcoded_world_to_ecef() -> np.ndarray:  # TODO remove when new dataset version is available
    """
    Return and hardcoded world_to_ecef matrix for dataset V1.0

    Returns:
        np.ndarray: 4x4 matrix
    """
    warnings.warn(
        "!!dataset metafile not found!! the hard-coded matrix will be loaded.\n"
        "This will be deprecated in future releases",
        PendingDeprecationWarning,
        stacklevel=3,
    )

    world_to_ecef = np.asarray(
        [
            [8.46617444e-01, 3.23463078e-01, -4.22623402e-01, -2.69876744e06],
            [-5.32201938e-01, 5.14559352e-01, -6.72301845e-01, -4.29315158e06],
            [-3.05311332e-16, 7.94103464e-01, 6.07782600e-01, 3.85516476e06],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        dtype=np.float64,
    )
    return world_to_ecef


def get_hardcoded_ecef_to_aerial() -> np.ndarray:  # TODO remove when new dataset version is available
    """
    Return and hardcoded ecef_to_aerial matrix for dataset V1.0

    Returns:
        np.ndarray: 4x4 matrix
    """
    warnings.warn(
        "!!dataset metafile not found!! the hard-coded matrix will be loaded.\n"
        "This will be deprecated in future releases",
        PendingDeprecationWarning,
        stacklevel=3,
    )

    ecef_to_aerial = np.asarray(
        [
            [-0.717416495, -1.14606296, -1.62854453, -572869.824],
            [1.80065798, -1.08914046, -0.0287877303, 300171.963],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ecef_to_aerial


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
    )

    filter_agents_threshold = raster_cfg["filter_agents_threshold"]
    history_num_frames = cfg["model_params"]["history_num_frames"]

    if map_type in ["py_satellite", "satellite_debug"]:
        sat_image = _load_satellite_map(raster_cfg["satellite_map_key"], data_manager)

        try:
            dataset_meta = _load_metadata(dataset_meta_key, data_manager)
            world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
            ecef_to_aerial = np.array(dataset_meta["ecef_to_aerial"], dtype=np.float64)

        except (KeyError, FileNotFoundError):  # TODO remove when new dataset version is available
            world_to_ecef = get_hardcoded_world_to_ecef()
            ecef_to_aerial = get_hardcoded_ecef_to_aerial()

        world_to_aerial = np.matmul(ecef_to_aerial, world_to_ecef)
        sat_rast = SatelliteRasterizer(render_context, sat_image, world_to_aerial)
        if map_type == "py_satellite":
            box_rast = BoxRasterizer(render_context, filter_agents_threshold, history_num_frames)
            return SatBoxRasterizer(
                render_context, filter_agents_threshold, history_num_frames, sat_rast=sat_rast, box_rast=box_rast,
            )
        else:
            return sat_rast

    elif map_type in ["py_semantic", "semantic_debug"]:
        semantic_map_filepath = data_manager.require(raster_cfg["semantic_map_key"])
        try:
            dataset_meta = _load_metadata(dataset_meta_key, data_manager)
            world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        except (KeyError, FileNotFoundError):  # TODO remove when new dataset version is available
            world_to_ecef = get_hardcoded_world_to_ecef()
        sem_rast = SemanticRasterizer(render_context, semantic_map_filepath, world_to_ecef)
        if map_type == "py_semantic":
            box_rast = BoxRasterizer(render_context, filter_agents_threshold, history_num_frames)
            return SemBoxRasterizer(
                render_context, filter_agents_threshold, history_num_frames, sem_rast=sem_rast, box_rast=box_rast,
            )
        else:
            return sem_rast

    elif map_type == "box_debug":
        return BoxRasterizer(render_context, filter_agents_threshold, history_num_frames)
    elif map_type == "stub_debug":
        return StubRasterizer(render_context, filter_agents_threshold)
    else:
        raise NotImplementedError(f"Rasterizer for map type {map_type} is not supported.")
