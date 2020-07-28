from typing import Sequence, no_type_check

import numpy as np
import pymap3d as pm

from ..geometry import transform_points
from .proto.road_network_pb2 import GeoFrame, Lane, MapFragment, TrafficControlElement


@no_type_check
def unpack_deltas_cm(
    dx: Sequence[int], dy: Sequence[int], dz: Sequence[int], g: GeoFrame, ecef_to_pose: np.ndarray
) -> np.ndarray:
    x = np.cumsum(np.asarray(dx) / 100)
    y = np.cumsum(np.asarray(dy) / 100)
    z = np.cumsum(np.asarray(dz) / 100)
    xyz = np.stack(pm.enu2ecef(x, y, z, g.origin.lat_e7 * 1e-7, g.origin.lng_e7 * 1e-7, 0), axis=-1)
    xyz = transform_points(xyz, ecef_to_pose)
    return xyz


@no_type_check
def unpack_boundary(b: Lane.Boundary, g: GeoFrame, ecef_to_pose: np.ndarray) -> np.ndarray:
    return unpack_deltas_cm(b.vertex_deltas_x_cm, b.vertex_deltas_y_cm, b.vertex_deltas_z_cm, g, ecef_to_pose)


@no_type_check
def unpack_crosswalk(e: TrafficControlElement, g: GeoFrame, ecef_to_pose: np.ndarray) -> np.ndarray:
    return unpack_deltas_cm(e.points_x_deltas_cm, e.points_y_deltas_cm, e.points_z_deltas_cm, g, ecef_to_pose)


@no_type_check
def proto_to_semantic_map(map_fragment: MapFragment, ecef_to_pose: np.ndarray) -> dict:
    """Loads and does preprocessing of given semantic map in binary proto format.

    Args:
        map_fragment (MapFragment): the external wrapper of the map's elements.
        ecef_to_pose (np.ndarray): ecef_to_pose matrix

    Returns:
        dict: A dict containing the semantic map contents.
    """

    # Unpack the semantic map. Right now we only extract position of lanes and crosswalks.
    lanes = []
    crosswalks = []

    lanes_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
    crosswalks_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]

    for element in map_fragment.elements:

        if element.element.HasField("lane"):
            lane = element.element.lane

            # get left-right coordinates and element id
            xyz_left = unpack_boundary(lane.left_boundary, lane.geo_frame, ecef_to_pose)
            xyz_right = unpack_boundary(lane.right_boundary, lane.geo_frame, ecef_to_pose)
            lanes.append({"xyz_left": xyz_left, "xyz_right": xyz_right, "id": element.id.id.decode("utf-8")})

            # store bounds for fast rasterisation look-up
            x_min = min(np.min(xyz_left[:, 0]), np.min(xyz_right[:, 0]))
            y_min = min(np.min(xyz_left[:, 1]), np.min(xyz_right[:, 1]))
            x_max = max(np.max(xyz_left[:, 0]), np.max(xyz_right[:, 0]))
            y_max = max(np.max(xyz_left[:, 1]), np.max(xyz_right[:, 1]))
            lanes_bounds = np.append(lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)

        if element.element.HasField("traffic_control_element"):
            traffic_element = element.element.traffic_control_element

            if traffic_element.HasField("pedestrian_crosswalk") and traffic_element.points_x_deltas_cm:
                xyz = unpack_crosswalk(traffic_element, traffic_element.geo_frame, ecef_to_pose)

                crosswalks.append({"xyz": xyz, "id": element.id.id.decode("utf-8")})

                crosswalks_bounds = np.append(
                    crosswalks_bounds,
                    np.asarray([[[np.min(xyz[:, 0]), np.min(xyz[:, 1])], [np.max(xyz[:, 0]), np.max(xyz[:, 1])]]]),
                    axis=0,
                )

    return {
        "lanes": lanes,
        "lanes_bounds": lanes_bounds,
        "crosswalks": crosswalks,
        "crosswalks_bounds": crosswalks_bounds,
    }
