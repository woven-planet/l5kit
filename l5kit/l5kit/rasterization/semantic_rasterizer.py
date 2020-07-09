from typing import List, Optional, Tuple

import cv2
import numpy as np
import pymap3d as pm

from ..geometry import rotation33_as_yaw, transform_point
from .rasterizer import Rasterizer


def render_semantic_map(
    semantic_map: dict, x: float, y: float, yaw: float, raster_size: Tuple[int, int], pixel_size: np.ndarray
) -> np.ndarray:
    """Renders the semantic map at given x,y coordinate with rotation ``yaw``.

    Args:
        semantic_map (dict): Semantic map dict.
        x (float): X coordinate to center crop from.
        y (float): Y coordinate to center crop from.
        yaw (float): Rotation of the cropped image.
        raster_size (Tuple[int, int]): Size of the image to render.
        pixel_size (np.ndarray): The scale of one pixel in the real world (in meters).

    Returns:
        np.ndarray: RGB raster.
    """

    img = 255 * np.ones(shape=(raster_size[1], raster_size[0], 3), dtype=np.uint8)

    def map_to_image(px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rx = (np.cos(yaw) * (px - x) - np.sin(yaw) * (py - y)) / pixel_size[0] + 0.5 * raster_size[0]
        ry = (np.sin(yaw) * (px - x) + np.cos(yaw) * (py - y)) / pixel_size[1] + 0.5 * raster_size[1]
        return rx.astype(int) * 256, ry.astype(int) * 256

    def elements_within_radius(bounding_box: dict, r: np.ndarray) -> np.ndarray:
        return np.nonzero(
            (x > bounding_box["min_x"] - r)
            & (x < bounding_box["max_x"] + r)
            & (y > bounding_box["min_y"] - r)
            & (y < bounding_box["max_y"] + r)
        )[0]

    r = np.linalg.norm(raster_size * pixel_size)

    # plot lanes
    lines = []
    for i in elements_within_radius(semantic_map["lanes_bounds"], r):
        lane = semantic_map["lanes"][i]
        x1, y1 = map_to_image(lane[0][0], lane[0][1])
        x2, y2 = map_to_image(lane[1][0], lane[1][1])
        pts1 = np.vstack((x1, y1)).T
        pts2 = np.vstack((x2, y2)).T
        lines.append(pts1)
        lines.append(pts2)
        poly = np.vstack((pts1, np.flip(pts2, 0)))
        cv2.fillPoly(img, [poly], (17, 17, 31), lineType=cv2.LINE_AA, shift=8)

    cv2.polylines(img, lines, False, (255, 217, 82), lineType=cv2.LINE_AA, shift=8)

    # plot crosswalks
    crosswalks = []
    for i in elements_within_radius(semantic_map["crosswalks_bounds"], r):
        crosswalk = semantic_map["crosswalks"][i]
        x1, y1 = map_to_image(crosswalk[0], crosswalk[1])
        pts = np.vstack((x1, y1)).T
        crosswalks.append(pts)

    cv2.polylines(img, crosswalks, True, (255, 117, 69), lineType=cv2.LINE_AA, shift=8)

    return img


class SemanticRasterizer(Rasterizer):
    """
    Rasteriser for the vectorised semantic map (generally loaded from json files).
    """

    def __init__(
        self,
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        ego_center: np.ndarray,
        semantic_map: dict,
        pose_to_ecef: np.ndarray,
    ):
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center

        self.pose_to_ecef = pose_to_ecef
        self.semantic_map = semantic_map

    def rasterize(
        self, history_frames: np.ndarray, history_agents: List[np.ndarray], agent: Optional[np.ndarray] = None
    ) -> np.ndarray:

        if agent is None:
            ego_translation = history_frames[0]["ego_translation"]
            ego_yaw = -rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw = -agent["yaw"]

        # move pose to enu frame the semantic map is stored in
        xyz = transform_point(ego_translation, self.pose_to_ecef)
        x, y, z = pm.ecef2enu(xyz[0], xyz[1], xyz[2], self.semantic_map["lat"], self.semantic_map["lon"], 0)

        # Apply ego_center offset
        # get_sat_image_crop_scaled crops around a point.
        # That point is ego_translation iff ego_center is [0.5, 0.5]
        # if not, we can compute by how much we need to translate in meters in the two directions and rotate the vector
        ego_offset = (0.5, 0.5) - np.asarray(self.ego_center)
        ego_offset_meters = np.asarray(self.pixel_size) * np.asarray(self.raster_size) * ego_offset

        rot_m = np.asarray([[np.cos(ego_yaw), np.sin(ego_yaw)], [-np.sin(ego_yaw), np.cos(ego_yaw)]])
        ego_offset_meters = rot_m @ ego_offset_meters

        sem_im = render_semantic_map(
            self.semantic_map,
            x + ego_offset_meters[0],
            y + ego_offset_meters[1],
            ego_yaw,
            raster_size=self.raster_size,
            pixel_size=self.pixel_size,
        )

        return sem_im.astype(np.float32) / 255

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)
