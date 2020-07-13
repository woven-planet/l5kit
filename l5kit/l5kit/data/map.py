from typing import Sequence, Tuple

import numpy as np
import pymap3d as pm

from .proto.road_network_pb2 import GeoFrame, Lane, MapFragment, TrafficControlElement


def load_semantic_map(semantic_map_path: str) -> dict:
    """Loads and does preprocessing of given semantic map in binary proto format.

    Args:
        semantic_map_path (str): The path of the semantic map file to load, a binary proto.

    Returns:
        dict: A dict containing the semantic map contents.
    """

    # store data in enu frame around origin lat, lon
    lat = 37.4108709
    lon = -122.1462192

    def unpack_deltas_cm(
        dx: Sequence[int], dy: Sequence[int], dz: Sequence[int], g: GeoFrame  # type: ignore
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.cumsum(np.asarray(dx) / 100)
        y = np.cumsum(np.asarray(dy) / 100)
        z = np.cumsum(np.asarray(dz) / 100)
        x, y, z = pm.enu2ecef(x, y, z, g.origin.lat_e7 * 1e-7, g.origin.lng_e7 * 1e-7, 0)  # type: ignore
        return pm.ecef2enu(x, y, z, lat, lon, 0)  # type:ignore

    def unpack_boundary(b: Lane.Boundary, g: GeoFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore
        return unpack_deltas_cm(b.vertex_deltas_x_cm, b.vertex_deltas_y_cm, b.vertex_deltas_z_cm, g)  # type: ignore

    def unpack_crosswalk(
        e: TrafficControlElement, g: GeoFrame  # type: ignore
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return unpack_deltas_cm(e.points_x_deltas_cm, e.points_y_deltas_cm, e.points_z_deltas_cm, g)  # type: ignore

    with open(semantic_map_path, "rb") as infile:
        mf = MapFragment()
        mf.ParseFromString(infile.read())

    # Unpack the semantic map. Right now we only extract position of lanes and crosswalks.
    lanes = []
    crosswalks = []

    lanes_min_x = []
    lanes_min_y = []
    lanes_max_x = []
    lanes_max_y = []

    crosswalks_min_x = []
    crosswalks_min_y = []
    crosswalks_max_x = []
    crosswalks_max_y = []

    for element in mf.elements:
        e = element.element

        if e.HasField("lane"):
            x1, y1, z1 = unpack_boundary(e.lane.left_boundary, e.lane.geo_frame)
            x2, y2, z2 = unpack_boundary(e.lane.right_boundary, e.lane.geo_frame)
            lanes.append(((x1, y1, z1), (x2, y2, z2)))

            lanes_min_x.append(min(np.min(x1), np.min(x2)))
            lanes_min_y.append(min(np.min(y1), np.min(y2)))
            lanes_max_x.append(max(np.max(x1), np.max(x2)))
            lanes_max_y.append(max(np.max(y1), np.max(y2)))

        if e.HasField("traffic_control_element"):
            tce = e.traffic_control_element
            if tce.HasField("pedestrian_crosswalk") and tce.points_x_deltas_cm:
                x, y, z = unpack_crosswalk(tce, tce.geo_frame)
                crosswalks.append((x, y, z))

                crosswalks_min_x.append(np.min(x))
                crosswalks_min_y.append(np.min(y))
                crosswalks_max_x.append(np.max(x))
                crosswalks_max_y.append(np.max(y))

    lane_bounds = {
        "min_x": np.array(lanes_min_x),
        "min_y": np.array(lanes_min_y),
        "max_x": np.array(lanes_max_x),
        "max_y": np.array(lanes_max_y),
    }

    crosswalks_bounds = {
        "min_x": np.array(crosswalks_min_x),
        "min_y": np.array(crosswalks_min_y),
        "max_x": np.array(crosswalks_max_x),
        "max_y": np.array(crosswalks_max_y),
    }

    return {
        "lat": lat,
        "lon": lon,
        "lanes": lanes,
        "lanes_bounds": lane_bounds,
        "crosswalks": crosswalks,
        "crosswalks_bounds": crosswalks_bounds,
    }
