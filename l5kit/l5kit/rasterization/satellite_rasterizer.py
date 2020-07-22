from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..geometry import rotation33_as_yaw, transform_point, world_to_image_pixels_matrix
from .rasterizer import Rasterizer
from .satellite_image import get_sat_image_crop_scaled


class SatelliteRasterizer(Rasterizer):
    """This rasterizer takes a satellite image in its constructor and a transform from world coordinates to this image.
    When you call rasterize, it will return a crop around the agent of interest with the agent's forward vector
    pointing right for the current timestep.
    """

    def __init__(
        self,
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        ego_center: np.ndarray,
        map_im: np.ndarray,
        map_to_sat: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """

        Arguments:
            raster_size (Tuple[int, int]): Desired output image size
            pixel_size (np.ndarray): Dimensions of one pixel in the real world
            ego_center (np.ndarray): Center of ego in the image, [0.5,0.5] would be in the image center.
            map_im (np.ndarray): Satellite image to crop from.
            map_to_sat (np.ndarray): Transform to go from map coordinates to satellite image pixel coordinates.
        """
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center
        self.map_im = map_im
        self.map_to_sat = map_to_sat
        self.interpolation = interpolation
        self.map_pixel_scale = (1 / np.linalg.norm(map_to_sat[0, 0:3]) + 1 / np.linalg.norm(map_to_sat[1, 0:3])) / 2

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tr_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if agent is None:
            ego_translation = history_frames[0]["ego_translation"]
            # Note 2: it looks like we are assuming that yaw in ecef == yaw in sat image
            ego_yaw = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            # Note 2: it looks like we are assuming that yaw in ecef == yaw in sat image
            ego_yaw = agent["yaw"]

        world_to_image_space = world_to_image_pixels_matrix(
            self.raster_size,
            self.pixel_size,
            ego_translation_m=ego_translation,
            ego_yaw_rad=ego_yaw,
            ego_center_in_image_ratio=self.ego_center,
        )

        # get the center of the images in meters using the inverse of the matrix,
        # Transform it to satellite coordinates (consider also z here)
        center_pixel = np.asarray(self.raster_size) * (0.5, 0.5)
        world_translation = transform_point(center_pixel, np.linalg.inv(world_to_image_space))
        sat_translation = transform_point(np.append(world_translation, ego_translation[2]), self.map_to_sat)

        # Note 1: there is a negation here, unknown why this is necessary.
        # My best guess is because Y is flipped, maybe we can do this more elegantly.
        sat_im = get_sat_image_crop_scaled(
            self.map_im,
            self.raster_size,
            sat_translation,
            yaw=-ego_yaw,
            pixel_size=self.pixel_size,
            sat_pixel_scale=self.map_pixel_scale,
            interpolation=self.interpolation,
        )

        # Here we flip the Y axis as Y+ should to the left of ego
        sat_im = sat_im[::-1]
        return sat_im.astype(np.float32) / 255

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)
