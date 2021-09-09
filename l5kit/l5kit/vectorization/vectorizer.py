from typing import Optional, List, Set, Dict

import numpy as np

from l5kit.data import (
    filter_agents_by_labels,
    filter_agents_by_distance,
    filter_tl_faces_by_status,
    PERCEPTION_LABEL_TO_INDEX,
)
from l5kit.sampling.agent_sampling import get_relative_poses
from l5kit.data.filter import filter_agents_by_track_id, get_other_agents_ids
from l5kit.data.map_api import CACHE_SIZE, InterpolationMethod
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from l5kit.geometry.transform import transform_points

class Vectorizer:
    def __init__(self, cfg, mapAPI):
        self.lane_cfg_params = cfg["data_generation_params"]["lane_params"]
        self.mapAPI = mapAPI
        self.max_agents_distance = cfg["data_generation_params"]["max_agents_distance"]
        self.history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"]
        self.future_num_frames=cfg["model_params"]["future_num_frames"]
        self.history_num_frames_max = max(cfg["model_params"]["history_num_frames_ego"], self.history_num_frames_agents)
        self.other_agents_num=cfg["data_generation_params"]["other_agents_num"]

    def vectorize(self, selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world, history_frames, history_agents, history_tl_faces, history_coords_offset, history_yaws_offset, history_availability, future_frames, future_agents):
        agent_dict = self._vectorize_agents(selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world, history_frames, history_agents, history_coords_offset, history_yaws_offset, history_availability, future_frames, future_agents)
        map_dict = self._vectorize_map(agent_centroid_m, agent_from_world, history_tl_faces)
        return {**agent_dict, **map_dict}

    def _vectorize_agents(self, selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world, history_frames, history_agents, history_coords_offset, history_yaws_offset, history_availability, future_frames, future_agents):
        # compute agent features
        # sequence_length x 2 (two being x, y)
        agent_points = history_coords_offset.copy()
        # sequence_length x 1
        agent_yaws = history_yaws_offset.copy()
        # sequence_length x xy+yaw (3)
        agent_trajectory_polyline = np.concatenate([agent_points, agent_yaws], axis=-1)
        agent_polyline_availability = history_availability.copy()

        # get agents around AoI sorted by distance in a given radius. Give priority to agents in the current time step
        history_agents_flat = filter_agents_by_labels(np.concatenate(history_agents))
        history_agents_flat = filter_agents_by_distance(history_agents_flat, agent_centroid_m, self.max_agents_distance)

        cur_agents = filter_agents_by_labels(history_agents[0])
        cur_agents = filter_agents_by_distance(cur_agents, agent_centroid_m, self.max_agents_distance)

        list_agents_to_take = get_other_agents_ids(
            history_agents_flat["track_id"], cur_agents["track_id"], selected_track_id, self.other_agents_num
        )

        # Loop to grab history and future for all other agents
        all_other_agents_history_positions = np.zeros((self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)
        all_other_agents_history_yaws = np.zeros((self.other_agents_num, self.history_num_frames_max + 1, 1), dtype=np.float32)
        all_other_agents_history_extents = np.zeros((self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)
        all_other_agents_history_availability = np.zeros((self.other_agents_num, self.history_num_frames_max + 1), dtype=np.float32)
        all_other_agents_types = np.zeros((self.other_agents_num,), dtype=np.int64)

        all_other_agents_future_positions = np.zeros((self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)
        all_other_agents_future_yaws = np.zeros((self.other_agents_num, self.future_num_frames, 1), dtype=np.float32)
        all_other_agents_future_extents = np.zeros((self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)
        all_other_agents_future_availability = np.zeros((self.other_agents_num, self.future_num_frames), dtype=np.float32)

        for idx, track_id in enumerate(list_agents_to_take):
            (
                agent_history_coords_offset,
                agent_history_yaws_offset,
                agent_history_extent,
                agent_history_availability,
            ) = get_relative_poses(
                self.history_num_frames_max + 1, history_frames, track_id, history_agents, agent_from_world, agent_yaw_rad
            )

            all_other_agents_history_positions[idx] = agent_history_coords_offset
            all_other_agents_history_yaws[idx] = agent_history_yaws_offset
            all_other_agents_history_extents[idx] = agent_history_extent
            all_other_agents_history_availability[idx] = agent_history_availability
            # NOTE (@lberg): assumption is that an agent doesn't change class (seems reasonable)
            # We look from history backward and choose the most recent time the track_id was available.
            current_other_actor = filter_agents_by_track_id(history_agents_flat, track_id)[0]
            all_other_agents_types[idx] = np.argmax(current_other_actor["label_probabilities"])

            (
                agent_future_coords_offset,
                agent_future_yaws_offset,
                agent_future_extent,
                agent_future_availability,
            ) = get_relative_poses(
                self.future_num_frames, future_frames, track_id, future_agents, agent_from_world, agent_yaw_rad
            )
            all_other_agents_future_positions[idx] = agent_future_coords_offset
            all_other_agents_future_yaws[idx] = agent_future_yaws_offset
            all_other_agents_future_extents[idx] = agent_future_extent
            all_other_agents_future_availability[idx] = agent_future_availability

        # crop similar to ego above
        all_other_agents_history_positions[:, self.history_num_frames_agents + 1 :] *= 0
        all_other_agents_history_yaws[:, self.history_num_frames_agents + 1 :] *= 0
        all_other_agents_history_extents[:, self.history_num_frames_agents + 1 :] *= 0
        all_other_agents_history_availability[:, self.history_num_frames_agents + 1 :] *= 0

        # compute other agents features
        # num_other_agents (M) x sequence_length x 2 (two being x, y)
        agents_points = all_other_agents_history_positions.copy()
        # num_other_agents (M) x sequence_length x 1
        agents_yaws = all_other_agents_history_yaws.copy()
        # agents_extents = all_other_agents_history_extents[:, :-1]
        # num_other_agents (M) x sequence_length x self._vector_length
        other_agents_polyline = np.concatenate([agents_points, agents_yaws], axis=-1)
        other_agents_polyline_availability = all_other_agents_history_availability.copy()

        agent_dict = {
            "all_other_agents_history_positions": all_other_agents_history_positions,
            "all_other_agents_history_yaws": all_other_agents_history_yaws,
            "all_other_agents_history_extents": all_other_agents_history_extents,
            "all_other_agents_history_availability": all_other_agents_history_availability.astype(np.bool),
            "all_other_agents_future_positions": all_other_agents_future_positions,
            "all_other_agents_future_yaws": all_other_agents_future_yaws,
            "all_other_agents_future_extents": all_other_agents_future_extents,
            "all_other_agents_future_availability": all_other_agents_future_availability.astype(np.bool),
            "all_other_agents_types": all_other_agents_types,
            "agent_trajectory_polyline": agent_trajectory_polyline,
            "agent_polyline_availability": agent_polyline_availability.astype(np.bool),
            "other_agents_polyline": other_agents_polyline,
            "other_agents_polyline_availability": other_agents_polyline_availability.astype(np.bool),
        }

        return agent_dict

    def _vectorize_map(self, agent_centroid_m, agent_from_world, history_tl_faces):    
        # START WORKING ON LANES
        # TODO (lberg): this implementation is super ugly, I'll fix it
        # TODO (anasrferreira): clean up more and add interpolation params to configuration as well.
        MAX_LANES = self.lane_cfg_params["max_num_lanes"]
        MAX_POINTS_LANES = self.lane_cfg_params["max_points_per_lane"]
        MAX_POINTS_CW = self.lane_cfg_params["max_points_per_crosswalk"]

        MAX_LANE_DISTANCE = self.lane_cfg_params["max_retrieval_distance_m"]
        INTERP_METHOD = InterpolationMethod.INTER_ENSURE_LEN  # split lane polyline by fixed number of points
        STEP_INTERPOLATION = MAX_POINTS_LANES  # number of points along lane
        MAX_CROSSWALKS = self.lane_cfg_params["max_num_crosswalks"]

        lanes_points = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES, 2), dtype=np.float32)
        lanes_availabilities = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES), dtype=np.float32)

        lanes_mid_points = np.zeros((MAX_LANES, MAX_POINTS_LANES, 2), dtype=np.float32)
        lanes_mid_availabilities = np.zeros((MAX_LANES, MAX_POINTS_LANES), dtype=np.float32)
        lanes_tl_feature = np.zeros((MAX_LANES, MAX_POINTS_LANES, 1), dtype=np.float32)

        # 8505 x 2 x 2
        lanes_bounds = self.mapAPI.bounds_info["lanes"]["bounds"]

        # filter first by bounds and then by distance, so that we always take the closest lanes
        lanes_indices = indices_in_bounds(agent_centroid_m, lanes_bounds, MAX_LANE_DISTANCE)
        distances = []
        for lane_idx in lanes_indices:
            lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
            lane = self.mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)
            lane_dist = np.linalg.norm(lane["xyz_midlane"][:, :2] - agent_centroid_m, axis=-1)
            distances.append(np.min(lane_dist))
        lanes_indices = lanes_indices[np.argsort(distances)]

        # TODO: move below after traffic lights
        crosswalks_bounds = self.mapAPI.bounds_info["crosswalks"]["bounds"]
        crosswalks_indices = indices_in_bounds(agent_centroid_m, crosswalks_bounds, MAX_LANE_DISTANCE)
        crosswalks_points = np.zeros((MAX_CROSSWALKS, MAX_POINTS_CW, 2), dtype=np.float32)
        crosswalks_availabilities = np.zeros_like(crosswalks_points[..., 0])
        for i, xw_idx in enumerate(crosswalks_indices[:MAX_CROSSWALKS]):
            xw_id = self.mapAPI.bounds_info["crosswalks"]["ids"][xw_idx]
            points = self.mapAPI.get_crosswalk_coords(xw_id)["xyz"]
            points = transform_points(points[:MAX_POINTS_CW, :2], agent_from_world)
            n = len(points)
            crosswalks_points[i, :n] = points
            crosswalks_availabilities[i, :n] = True

        active_tl_faces = set(filter_tl_faces_by_status(history_tl_faces[0], "ACTIVE")["face_id"].tolist())
        active_tl_face_to_color: Dict[str, str] = {}
        for face in active_tl_faces:
            try:
                active_tl_face_to_color[face] = self.mapAPI.get_color_for_face(face).lower() # TODO: why lower()?
            except KeyError:
                continue  # this happens only on KIRBY, 2 TLs have no match in the map

        for out_idx, lane_idx in enumerate(lanes_indices[:MAX_LANES]):
            lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
            lane = self.mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)

            xy_left = lane["xyz_left"][:MAX_POINTS_LANES, :2]
            xy_right = lane["xyz_right"][:MAX_POINTS_LANES, :2]
            # convert coordinates into local space
            xy_left = transform_points(xy_left, agent_from_world)
            xy_right = transform_points(xy_right, agent_from_world)

            num_vectors_left = len(xy_left)
            num_vectors_right = len(xy_right)

            lanes_points[out_idx * 2, :num_vectors_left] = xy_left
            lanes_points[out_idx * 2 + 1, :num_vectors_right] = xy_right

            lanes_availabilities[out_idx * 2, :num_vectors_left] = 1
            lanes_availabilities[out_idx * 2 + 1, :num_vectors_right] = 1

            midlane = lane["xyz_midlane"][:MAX_POINTS_LANES, :2]
            midlane = transform_points(midlane, agent_from_world)
            num_vectors_mid = len(midlane)

            lanes_mid_points[out_idx, :num_vectors_mid] = midlane
            lanes_mid_availabilities[out_idx, :num_vectors_mid] = 1

            lanes_tl_feature[out_idx, :num_vectors_mid] = self.mapAPI.get_tl_feature_for_lane(lane_id, active_tl_face_to_color)

        # disable all points over the distance threshold
        valid_distances = np.linalg.norm(lanes_points, axis=-1) < MAX_LANE_DISTANCE
        lanes_availabilities *= valid_distances
        valid_mid_distances = np.linalg.norm(lanes_mid_points, axis=-1) < MAX_LANE_DISTANCE
        lanes_mid_availabilities *= valid_mid_distances

        # 2 MAX_LANES x MAX_VECTORS x (XY + TL-feature)
        # -> 2 MAX_LANES for left and right
        lanes = np.concatenate([lanes_points, np.zeros_like(lanes_points[..., [0]])], axis=-1)
        # pad such that length is 3
        crosswalks = np.concatenate([crosswalks_points, np.zeros_like(crosswalks_points[..., [0]])], axis=-1)
        # MAX_LANES x MAX_VECTORS x 3 (XY + 1 TL-feature)
        lanes_mid = np.concatenate([lanes_mid_points, lanes_tl_feature], axis=-1)

        return {
            "lanes": lanes,
            "lanes_availabilities": lanes_availabilities.astype(np.bool),
            "lanes_mid": lanes_mid,
            "lanes_mid_availabilities": lanes_mid_availabilities.astype(np.bool),
            "crosswalks": crosswalks,
            "crosswalks_availabilities": crosswalks_availabilities.astype(np.bool),
        }