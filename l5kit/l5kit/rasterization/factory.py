import json
import os
from typing import Tuple, cast

import cv2
import numpy as np

from ..data import DataManager, load_pose_to_ecef, load_semantic_map
from .rasterizer import Rasterizer
from .sat_box_rasterizer import SatBoxRasterizer
from .sem_box_rasterizer import SemBoxRasterizer


def _load_image_and_metadata(image_key: str, data_manager: DataManager) -> Tuple[np.ndarray, dict]:
    """Loads image from given key and its meatadata. The metadata file should be a file with the same key except for
    having a .json extension instead.

    Args:
        image_key (str): key to the image (e.g. ``maps/my_satellite_image.png``)
        data_manager (DataManager): DataManager used for requiring files

    Raises:
        Exception: Image or metadata is missing or invalid

    Returns:
        Tuple[np.ndarray, dict]: Image and metadata
    """

    image_metadata_key = os.path.splitext(image_key)[0] + ".json"
    image_path = data_manager.require(image_key)
    image_metadata_path = data_manager.require(image_metadata_key)

    image = cv2.imread(image_path)[..., ::-1]  # BGR->RGB
    if image is None:
        raise Exception(f"Failed to load image from {image_path}")

    with open(image_metadata_path, "r") as f:
        metadata = json.load(f)

    return image, metadata


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

    raster_size: Tuple[int, int] = cast(Tuple[int, int], tuple(raster_cfg["raster_size"]))
    pixel_size = np.array(raster_cfg["pixel_size"])
    ego_center = np.array(raster_cfg["ego_center"])
    filter_agents_threshold = raster_cfg["filter_agents_threshold"]

    if map_type in ["py_satellite", "satellite_rgb"]:
        sat_image, meta = _load_image_and_metadata(raster_cfg["satellite_map_key"], data_manager)
        ecef_to_sat = np.array(meta["ecef_to_image"], dtype=np.float64)
        pose_to_ecef = load_pose_to_ecef()

        map_to_sat = np.matmul(ecef_to_sat, pose_to_ecef)
        return SatBoxRasterizer(raster_size, pixel_size, ego_center, filter_agents_threshold, sat_image, map_to_sat)
    elif map_type == "py_semantic":
        semantic_map_filepath = data_manager.require(raster_cfg["semantic_map_key"])
        semantic_map = load_semantic_map(semantic_map_filepath)
        pose_to_ecef = load_pose_to_ecef()

        return SemBoxRasterizer(
            raster_size, pixel_size, ego_center, filter_agents_threshold, semantic_map, pose_to_ecef
        )
    else:
        raise NotImplementedError(f"Rasterizer for map type {map_type} is not supported.")
