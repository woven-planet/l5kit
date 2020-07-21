from typing import Sequence, no_type_check

import numpy as np
import pymap3d as pm

from .proto.road_network_pb2 import GeoFrame, Lane, MapFragment, TrafficControlElement

# store data in enu frame around origin lat, lon
LAT = 37.4108709
LON = -122.1462192


@no_type_check
def unpack_deltas_cm(dx: Sequence[int], dy: Sequence[int], dz: Sequence[int], g: GeoFrame) -> np.ndarray:
    x = np.cumsum(np.asarray(dx) / 100)
    y = np.cumsum(np.asarray(dy) / 100)
    z = np.cumsum(np.asarray(dz) / 100)
    x, y, z = pm.enu2ecef(x, y, z, g.origin.lat_e7 * 1e-7, g.origin.lng_e7 * 1e-7, 0)
    x, y, z = pm.ecef2enu(x, y, z, LAT, LON, 0)
    return np.stack([x, y, z])


@no_type_check
def unpack_boundary(b: Lane.Boundary, g: GeoFrame) -> np.ndarray:
    return unpack_deltas_cm(b.vertex_deltas_x_cm, b.vertex_deltas_y_cm, b.vertex_deltas_z_cm, g)


@no_type_check
def unpack_crosswalk(e: TrafficControlElement, g: GeoFrame) -> np.ndarray:
    return unpack_deltas_cm(e.points_x_deltas_cm, e.points_y_deltas_cm, e.points_z_deltas_cm, g)


@no_type_check
def proto_to_semantic_map(map_fragment: "MapFragment") -> dict:
    """Loads and does preprocessing of given semantic map in binary proto format.

    Args:
        map_fragment (MapFragment): the external wrapper of the map's elements.

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
            x_left, y_left, z_left = unpack_boundary(lane.left_boundary, lane.geo_frame)
            x_right, y_right, z_right = unpack_boundary(lane.right_boundary, lane.geo_frame)
            lanes.append(
                {
                    "xyz_left": (x_left, y_left, z_left),
                    "xyz_right": (x_right, y_right, z_right),
                    "id": element.id.id.decode("utf-8"),
                }
            )
            # store bounds for fast rasterisation look-up
            x_min = min(np.min(x_left), np.min(x_right))
            y_min = min(np.min(y_left), np.min(y_right))
            x_max = max(np.max(x_left), np.max(x_right))
            y_max = max(np.max(y_left), np.max(y_right))
            lanes_bounds = np.append(lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)

        if element.element.HasField("traffic_control_element"):
            traffic_element = element.element.traffic_control_element

            if traffic_element.HasField("pedestrian_crosswalk") and traffic_element.points_x_deltas_cm:
                x, y, z = unpack_crosswalk(traffic_element, traffic_element.geo_frame)

                crosswalks.append({"xyz": (x, y, z), "id": element.id.id.decode("utf-8")})

                crosswalks_bounds = np.append(
                    crosswalks_bounds, np.asarray([[[np.min(x), np.min(y)], [np.max(x), np.max(y)]]]), axis=0
                )

    return {
        "lat": LAT,
        "lon": LON,
        "lanes": lanes,
        "lanes_bounds": lanes_bounds,
        "crosswalks": crosswalks,
        "crosswalks_bounds": crosswalks_bounds,
    }
