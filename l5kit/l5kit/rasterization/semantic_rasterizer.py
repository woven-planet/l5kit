from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..data import proto_to_semantic_map
from ..data.proto.road_network_pb2 import MapFragment
from ..geometry import rotation33_as_yaw, transform_point, transform_points, world_to_image_pixels_matrix
from .rasterizer import Rasterizer


def elements_within_radius(x: float, y: float, bounds: np.ndarray, radius: float) -> np.ndarray:
    """
    Get indices of elements for which bounds are inside a radius from center (x,y)

    Args:
        x (float): center x
        y (float): center y
        bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
        radius (float): radius value

    Returns:
        np.ndarray: indices of elements inside radius from center
    """
    x_min_in = x > bounds[:, 0, 0] - radius
    y_min_in = y > bounds[:, 0, 1] - radius
    x_max_in = x < bounds[:, 1, 0] + radius
    y_max_in = y < bounds[:, 1, 1] + radius
    return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]


class SemanticRasterizer(Rasterizer):
    """
    Rasteriser for the vectorised semantic map (generally loaded from json files).
    """

    def __init__(
        self,
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        ego_center: np.ndarray,
        semantic_map_path: str,
        pose_to_ecef: np.ndarray,
    ):
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center

        self.pose_to_ecef = pose_to_ecef

        # load protobuf and process it, but keep also the original proto elements
        with open(semantic_map_path, "rb") as infile:
            mapfrag = MapFragment()
            mapfrag.ParseFromString(infile.read())

        ecef_to_pose = np.linalg.inv(self.pose_to_ecef)

        self.semantic_map = proto_to_semantic_map(mapfrag, ecef_to_pose)
        self.elements_lookup = {el.id.id.decode("utf-8"): el for el in mapfrag.elements}

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tr_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # TODO TR_FACES

        if agent is None:
            ego_translation = history_frames[0]["ego_translation"]
            ego_yaw = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw = agent["yaw"]

        world_to_image_space = world_to_image_pixels_matrix(
            self.raster_size, self.pixel_size, ego_translation, ego_yaw, self.ego_center,
        )

        # get pose of center pixel
        center_pixel = np.asarray(self.raster_size) * (0.5, 0.5)
        world_translation = transform_point(center_pixel, np.linalg.inv(world_to_image_space))
        xyz = np.append(world_translation, ego_translation[2])

        # TODO (lberg): understand -yaw here in relation with map_to_image
        sem_im = self.render_semantic_map(xyz[0], xyz[1], world_to_image_space)
        return sem_im.astype(np.float32) / 255

    def render_semantic_map(self, x: float, y: float, world_to_image_space: np.ndarray) -> np.ndarray:
        """Renders the semantic map at given x,y coordinate with rotation ``yaw``.

        Args:
            x (float): X coordinate to center crop from.
            y (float): Y coordinate to center crop from.
            world_to_image_space (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """

        img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)
        r = float(np.linalg.norm(self.raster_size * self.pixel_size))

        # plot lanes
        lines = []
        for idx in elements_within_radius(x, y, self.semantic_map["lanes_bounds"], r):
            lane = self.semantic_map["lanes"][idx]
            xy_left = transform_points(lane["xyz_left"][:, :2], world_to_image_space).astype(np.int) * 256
            xy_right = transform_points(lane["xyz_right"][:, :2], world_to_image_space).astype(np.int) * 256

            lines.append(xy_left)
            lines.append(xy_right)
            poly = np.vstack((xy_left, np.flip(xy_right, 0)))
            cv2.fillPoly(img, [poly], (17, 17, 31), lineType=cv2.LINE_AA, shift=8)

        cv2.polylines(img, lines, False, (255, 217, 82), lineType=cv2.LINE_AA, shift=8)

        # plot crosswalks
        crosswalks = []
        for idx in elements_within_radius(x, y, self.semantic_map["crosswalks_bounds"], r):
            crosswalk = self.semantic_map["crosswalks"][idx]
            xy_cross = transform_points(crosswalk["xyz"][:, :2], world_to_image_space).astype(np.int) * 256
            crosswalks.append(xy_cross)

        cv2.polylines(img, crosswalks, True, (255, 117, 69), lineType=cv2.LINE_AA, shift=8)

        return img

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)
