import json
import os
from typing import Tuple, cast

import cv2
import numpy as np

from ..data import DataManager
from .box_rasterizer import BoxRasterizer
from .rasterizer import Rasterizer
from .sat_box_rasterizer import SatBoxRasterizer
from .satellite_rasterizer import SatelliteRasterizer
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

    raster_size: Tuple[int, int] = cast(Tuple[int, int], tuple(raster_cfg["raster_size"]))
    pixel_size = np.array(raster_cfg["pixel_size"])
    ego_center = np.array(raster_cfg["ego_center"])
    filter_agents_threshold = raster_cfg["filter_agents_threshold"]
    history_num_frames = cfg["model_params"]["history_num_frames"]

    if map_type == "py_satellite":
        sat_image_key = raster_cfg["satellite_map_key"]
        sat_meta_key = os.path.splitext(sat_image_key)[0] + ".json"

        sat_image = _load_satellite_map(sat_image_key, data_manager)
        sat_meta = _load_metadata(sat_meta_key, data_manager)
        ecef_to_sat = np.array(sat_meta["ecef_to_image"], dtype=np.float64)

        try:
            dataset_meta = _load_metadata(dataset_meta_key, data_manager)
            pose_to_ecef = np.array(dataset_meta["pose_to_ecef"], dtype=np.float64)
        except (KeyError, FileNotFoundError):  # TODO remove when new dataset version is available
            print(
                "!!dataset metafile not found!! the hard-coded matrix will be loaded.\n"
                "This will be deprecated in future releases"
            )
            pose_to_ecef = np.asarray(
                [
                    [8.46617444e-01, 3.23463078e-01, -4.22623402e-01, -2.69876744e06],
                    [-5.32201938e-01, 5.14559352e-01, -6.72301845e-01, -4.29315158e06],
                    [-3.05311332e-16, 7.94103464e-01, 6.07782600e-01, 3.85516476e06],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ],
                dtype=np.float64,
            )

        map_to_sat = np.matmul(ecef_to_sat, pose_to_ecef)
        return SatBoxRasterizer(
            raster_size, pixel_size, ego_center, filter_agents_threshold, history_num_frames, sat_image, map_to_sat
        )
    elif map_type == "py_semantic":
        semantic_map_path = data_manager.require(raster_cfg["semantic_map_key"])
        try:
            dataset_meta = _load_metadata(dataset_meta_key, data_manager)
            pose_to_ecef = np.array(dataset_meta["pose_to_ecef"], dtype=np.float64)
        except (KeyError, FileNotFoundError):  # TODO remove when new dataset version is available
            print(
                "!!dataset metafile not found!! the hard-coded matrix will be loaded.\n"
                "This will be deprecated in future releases"
            )
            pose_to_ecef = np.asarray(
                [
                    [8.46617444e-01, 3.23463078e-01, -4.22623402e-01, -2.69876744e06],
                    [-5.32201938e-01, 5.14559352e-01, -6.72301845e-01, -4.29315158e06],
                    [-3.05311332e-16, 7.94103464e-01, 6.07782600e-01, 3.85516476e06],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ],
                dtype=np.float64,
            )

        return SemBoxRasterizer(
            raster_size,
            pixel_size,
            ego_center,
            filter_agents_threshold,
            history_num_frames,
            semantic_map_path,
            pose_to_ecef,
        )
    elif map_type == "box_debug":
        return BoxRasterizer(raster_size, pixel_size, ego_center, filter_agents_threshold, history_num_frames)
    elif map_type == "satellite_debug":
        sat_image_key = raster_cfg["satellite_map_key"]
        sat_meta_key = os.path.splitext(sat_image_key)[0] + ".json"

        sat_image = _load_satellite_map(sat_image_key, data_manager)
        sat_meta = _load_metadata(sat_meta_key, data_manager)
        ecef_to_sat = np.array(sat_meta["ecef_to_image"], dtype=np.float64)

        try:
            dataset_meta = _load_metadata(dataset_meta_key, data_manager)
            pose_to_ecef = np.array(dataset_meta["pose_to_ecef"], dtype=np.float64)
        except (KeyError, FileNotFoundError):  # TODO remove when new dataset version is available
            print(
                "!!dataset metafile not found!! the hard-coded matrix will be loaded.\n"
                "This will be deprecated in future releases"
            )
            pose_to_ecef = np.asarray(
                [
                    [8.46617444e-01, 3.23463078e-01, -4.22623402e-01, -2.69876744e06],
                    [-5.32201938e-01, 5.14559352e-01, -6.72301845e-01, -4.29315158e06],
                    [-3.05311332e-16, 7.94103464e-01, 6.07782600e-01, 3.85516476e06],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ],
                dtype=np.float64,
            )

        map_to_sat = np.matmul(ecef_to_sat, pose_to_ecef)
        return SatelliteRasterizer(raster_size, pixel_size, ego_center, sat_image, map_to_sat)
    else:
        raise NotImplementedError(f"Rasterizer for map type {map_type} is not supported.")
