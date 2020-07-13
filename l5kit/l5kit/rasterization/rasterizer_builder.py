import json
import os
from typing import Tuple, cast

import cv2
import numpy as np

from ..data import DataManager, load_semantic_map
from .rasterizer import Rasterizer
from .sat_box_rasterizer import SatBoxRasterizer
from .sem_box_rasterizer import SemBoxRasterizer


def _load_metadata(meta_key: str, data_manager: DataManager) -> dict:
    """
    Load a json metadata file

    Args:
        meta_key (str): relative key to the metadata
        data_manager (DataManager): DataManager used for requiring files

    Returns:
        dict: metadata as a dict
    """
    image_metadata_path = data_manager.require(meta_key)
    with open(image_metadata_path, "r") as f:
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
    data_meta_key = raster_cfg["dataset_meta_key"]

    raster_size: Tuple[int, int] = cast(Tuple[int, int], tuple(raster_cfg["raster_size"]))
    pixel_size = np.array(raster_cfg["pixel_size"])
    ego_center = np.array(raster_cfg["ego_center"])
    filter_agents_threshold = raster_cfg["filter_agents_threshold"]
    history_num_frames = cfg["model_params"]["history_num_frames"]

    if map_type in ["py_satellite", "satellite_rgb"]:
        sat_image_key = raster_cfg["satellite_map_key"]
        sat_image = _load_satellite_map(sat_image_key, data_manager)

        sat_meta_key = os.path.splitext(sat_image_key)[0] + ".json"
        ecef_to_sat = np.array(_load_metadata(sat_meta_key, data_manager)["ecef_to_image"], dtype=np.float64)
        try:
            pose_to_ecef = np.array(_load_metadata(data_meta_key, data_manager)["pose_to_ecef"], dtype=np.float64)
        except FileNotFoundError:  # TODO remove in v1.0.5
            raise FileNotFoundError(
                "!!dataset metafile not found!! this check has been introduced in l5kit v1.0.4\n"
                "The file is already available in the public dataset folder, please download it.\n"
                "This message will be removed in l5kit v1.0.5"
            )

        map_to_sat = np.matmul(ecef_to_sat, pose_to_ecef)
        return SatBoxRasterizer(
            raster_size, pixel_size, ego_center, filter_agents_threshold, history_num_frames, sat_image, map_to_sat
        )
    elif map_type == "py_semantic":
        semantic_map_filepath = data_manager.require(raster_cfg["semantic_map_key"])
        semantic_map = load_semantic_map(semantic_map_filepath)
        try:
            pose_to_ecef = np.array(_load_metadata(data_meta_key, data_manager)["pose_to_ecef"], dtype=np.float64)
        except FileNotFoundError:  # TODO remove in v1.0.5
            raise FileNotFoundError(
                "!!dataset metafile not found!! this check has been introduced in l5kit v1.0.4\n"
                "The file is already available in the public dataset folder, please download it.\n"
                "This message will be removed in l5kit v1.0.5"
            )

        return SemBoxRasterizer(
            raster_size,
            pixel_size,
            ego_center,
            filter_agents_threshold,
            history_num_frames,
            semantic_map,
            pose_to_ecef,
        )
    else:
        raise NotImplementedError(f"Rasterizer for map type {map_type} is not supported.")
