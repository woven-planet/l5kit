from collections import defaultdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..data.filter import get_tl_faces_active
from ..data.map_api import MapAPI
from ..geometry import rotation33_as_yaw, transform_point, transform_points, world_to_image_pixels_matrix
from .rasterizer import Rasterizer

# sub-pixel drawing precision constants
CV2_SHIFT = 8  # how many bits to shift in drawing
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT


def elements_within_bounds(center: np.ndarray, bounds: np.ndarray, half_extent: float) -> np.ndarray:
    """
    Get indices of elements for which the bounding box described by bounds intersects the one defined around
    center (square with side 2*half_side)

    Args:
        center (float): XY of the center
        bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
        half_extent (float): half the side of the bounding box centered around center

    Returns:
        np.ndarray: indices of elements inside radius from center
    """
    x_center, y_center = center

    x_min_in = x_center > bounds[:, 0, 0] - half_extent
    y_min_in = y_center > bounds[:, 0, 1] - half_extent
    x_max_in = x_center < bounds[:, 1, 0] + half_extent
    y_max_in = y_center < bounds[:, 1, 1] + half_extent
    return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]


def cv2_subpixel(coords: np.ndarray) -> np.ndarray:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT
    cv2 calls will use shift to restore original values with higher precision

    Args:
        coords (np.ndarray): XY coords as float

    Returns:
        np.ndarray: XY coords as int for cv2 shift draw
    """
    coords = coords * CV2_SHIFT_VALUE
    coords = coords.astype(np.int)
    return coords


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

        self.proto_API = MapAPI(semantic_map_path, pose_to_ecef)

        self.bounds_info = self.get_bounds()

    # TODO is this the right place for this function?
    def get_bounds(self) -> dict:
        """
        For each elements of interest returns bounds [[min_x, min_y],[max_x, max_y]] and proto ids

        Returns:
            dict: keys are classes of elements, values are dict with `bounds` and `ids` keys
        """
        lanes_ids = []
        crosswalks_ids = []

        lanes_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
        crosswalks_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]

        for element in self.proto_API:
            element_id = MapAPI.id_as_str(element.id)

            if self.proto_API.is_lane(element):
                lane = self.proto_API.get_lane_coords(element_id)
                # store bounds for fast rasterisation look-up
                x_min = min(np.min(lane["xyz_left"][:, 0]), np.min(lane["xyz_right"][:, 0]))
                y_min = min(np.min(lane["xyz_left"][:, 1]), np.min(lane["xyz_right"][:, 1]))
                x_max = max(np.max(lane["xyz_left"][:, 0]), np.max(lane["xyz_right"][:, 0]))
                y_max = max(np.max(lane["xyz_left"][:, 1]), np.max(lane["xyz_right"][:, 1]))

                lanes_bounds = np.append(lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
                lanes_ids.append(element_id)

            if self.proto_API.is_crosswalk(element):
                crosswalk = self.proto_API.get_crosswalk_coords(element_id)
                # store bounds for fast rasterisation look-up
                x_min = np.min(crosswalk["xyz"][:, 0])
                y_min = np.min(crosswalk["xyz"][:, 1])
                x_max = np.max(crosswalk["xyz"][:, 0])
                y_max = np.max(crosswalk["xyz"][:, 1])

                crosswalks_bounds = np.append(
                    crosswalks_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0,
                )
                crosswalks_ids.append(element_id)

        return {
            "lanes": {"bounds": lanes_bounds, "ids": lanes_ids},
            "crosswalks": {"bounds": crosswalks_bounds, "ids": crosswalks_ids},
        }

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
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

        # get XY of center pixel in world coordinates
        center_pixel = np.asarray(self.raster_size) * (0.5, 0.5)
        center_world = transform_point(center_pixel, np.linalg.inv(world_to_image_space))

        sem_im = self.render_semantic_map(center_world, world_to_image_space, history_tl_faces[0])
        return sem_im.astype(np.float32) / 255

    def render_semantic_map(
        self, center_world: np.ndarray, world_to_image_space: np.ndarray, tl_faces: np.ndarray
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_world (np.ndarray): XY of the image center in world ref system
            world_to_image_space (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """

        img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(get_tl_faces_active(tl_faces)["gid"].tolist())

        # plot lanes
        lanes_lines = defaultdict(list)

        for idx in elements_within_bounds(center_world, self.bounds_info["lanes"]["bounds"], raster_radius):
            lane = self.proto_API[self.bounds_info["lanes"]["ids"][idx]].element.lane

            # get image coords
            lane_coords = self.proto_API.get_lane_coords(self.bounds_info["lanes"]["ids"][idx])
            xy_left = cv2_subpixel(transform_points(lane_coords["xyz_left"][:, :2], world_to_image_space))
            xy_right = cv2_subpixel(transform_points(lane_coords["xyz_right"][:, :2], world_to_image_space))
            lanes_area = np.vstack((xy_left, np.flip(xy_right, 0)))  # start->end left then end->start right

            # Note(lberg): this called on all polygons skips some of them, don't know why
            cv2.fillPoly(img, [lanes_area], (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

            lane_type = "default"  # no traffic light face is controlling this lane
            # TODO should we change the next lane (lanes_ahead) status?
            lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                if self.proto_API.is_traffic_face_red(tl_id):
                    lane_type = "red"
                elif self.proto_API.is_traffic_face_green(tl_id):
                    lane_type = "green"
                elif self.proto_API.is_traffic_face_yellow(tl_id):
                    lane_type = "yellow"

            lanes_lines[lane_type].extend([xy_left, xy_right])

        cv2.polylines(img, lanes_lines["default"], False, (255, 217, 82), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines["green"], False, (0, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines["yellow"], False, (255, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines["red"], False, (255, 0, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        # plot crosswalks
        crosswalks = []
        for idx in elements_within_bounds(center_world, self.bounds_info["crosswalks"]["bounds"], raster_radius):
            crosswalk = self.proto_API.get_crosswalk_coords(self.bounds_info["crosswalks"]["ids"][idx])

            xy_cross = cv2_subpixel(transform_points(crosswalk["xyz"][:, :2], world_to_image_space))
            crosswalks.append(xy_cross)

        cv2.polylines(img, crosswalks, True, (255, 117, 69), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        return img

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)
