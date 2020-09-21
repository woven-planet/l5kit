from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..geometry import rotation33_as_yaw, transform_point
from .rasterizer import Rasterizer
from .satellite_image import get_sat_image_crop_scaled
from .render_context import RenderContext


class SatelliteRasterizer(Rasterizer):
    """This rasterizer takes a satellite image in its constructor and a transform from world coordinates to this image.
    When you call rasterize, it will return a crop around the agent of interest with the agent's forward vector
    pointing right for the current timestep.
    """

    def __init__(
        self,
        render_context: RenderContext,
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        map_im: np.ndarray,
        world_to_aerial: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """

        Arguments:
            raster_size (Tuple[int, int]): Desired output image size
            pixel_size (np.ndarray): Dimensions of one pixel in the real world
            ego_center (np.ndarray): Center of ego in the image, [0.5,0.5] would be in the image center.
            map_im (np.ndarray): Satellite image to crop from.
            world_to_aerial (np.ndarray): Transform to go from map coordinates to satellite image pixel coordinates.
        """
        self.render_context = render_context
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.map_im = map_im
        self.world_to_aerial = world_to_aerial
        self.interpolation = interpolation
        self.map_pixel_scale = (
            1 / np.linalg.norm(world_to_aerial[0, 0:3]) + 1 / np.linalg.norm(world_to_aerial[1, 0:3])
        ) / 2

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            # Note 2: it looks like we are assuming that yaw in ecef == yaw in sat image
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])

        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            # Note 2: it looks like we are assuming that yaw in ecef == yaw in sat image
            ego_yaw_rad = agent["yaw"]

        # Compute ego pose from its position and heading
        ego_pose = np.array([[np.cos(ego_yaw_rad), -np.sin(ego_yaw_rad), ego_translation_m[0]],
                             [np.sin(ego_yaw_rad), np.cos(ego_yaw_rad), ego_translation_m[1]],
                             [0, 0, 1]])

        ego_from_global = np.linalg.inv(ego_pose)
        raster_from_global = self.render_context.raster_from_local @ ego_from_global
        global_from_raster = np.linalg.inv(raster_from_global)

        # Transform raster center to satellite coordinates (consider also z here)
        raster_center_px = np.asarray(self.raster_size) * (0.5, 0.5)
        raster_center_in_global = transform_point(raster_center_px, global_from_raster)
        raster_center_in_sat = transform_point(np.append(raster_center_in_global, ego_translation_m[2]), self.world_to_aerial)

        # Note 1: there is a negation here, unknown why this is necessary.
        # My best guess is because Y is flipped, maybe we can do this more elegantly.
        sat_im = get_sat_image_crop_scaled(
            self.map_im,
            self.raster_size,
            raster_center_in_sat,
            yaw=-ego_yaw_rad,
            pixel_size=self.pixel_size,
            sat_pixel_scale=self.map_pixel_scale,
            interpolation=self.interpolation,
        )

        # Here we flip the Y axis as Y+ should to the left of ego
        sat_im = sat_im[::-1]
        return sat_im.astype(np.float32) / 255

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)
