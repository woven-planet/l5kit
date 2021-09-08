
from typing import Dict, List, Optional, Set
from functools import lru_cache, partial
from l5kit.geometry.transform import transform_points

import numpy as np

from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.dataset.agent import MIN_FRAME_FUTURE, MIN_FRAME_HISTORY

from l5kit.sampling.agent_sampling import get_relative_poses, get_agent_context, compute_agent_velocity
from l5kit.data import (
    ChunkedDataset,
    filter_agents_by_labels,
    filter_tl_faces_by_status,
    PERCEPTION_LABEL_TO_INDEX,
)

from l5kit.data.filter import filter_agents_by_track_id
from l5kit.geometry import compute_agent_pose, rotation33_as_yaw
from l5kit.rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, RenderContext
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from l5kit.data.map_api import CACHE_SIZE, InterpolationMethod, MapAPI

def agents_func(history_coords_offset, history_yaws_offset, history_availability, history_agents, agent_centroid_m, max_agents_distance, selected_track_id, other_agents_num, history_num_frames_max, future_num_frames, history_frames, agent_from_world, agent_yaw_rad, future_frames, future_agents, history_num_frames_agents):
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
    history_agents_flat = filter_agents_by_distance(history_agents_flat, agent_centroid_m, max_agents_distance)

    cur_agents = filter_agents_by_labels(history_agents[0])
    cur_agents = filter_agents_by_distance(cur_agents, agent_centroid_m, max_agents_distance)

    list_agents_to_take = get_other_agents_ids(
        history_agents_flat["track_id"], cur_agents["track_id"], selected_track_id, other_agents_num
    )

    # Loop to grab history and future for all other agents
    all_other_agents_history_positions = np.zeros((other_agents_num, history_num_frames_max + 1, 2), dtype=np.float32)
    all_other_agents_history_yaws = np.zeros((other_agents_num, history_num_frames_max + 1, 1), dtype=np.float32)
    all_other_agents_history_extents = np.zeros((other_agents_num, history_num_frames_max + 1, 2), dtype=np.float32)
    all_other_agents_history_availability = np.zeros((other_agents_num, history_num_frames_max + 1), dtype=np.float32)
    all_other_agents_types = np.zeros((other_agents_num,), dtype=np.int64)

    all_other_agents_future_positions = np.zeros((other_agents_num, future_num_frames, 2), dtype=np.float32)
    all_other_agents_future_yaws = np.zeros((other_agents_num, future_num_frames, 1), dtype=np.float32)
    all_other_agents_future_extents = np.zeros((other_agents_num, future_num_frames, 2), dtype=np.float32)
    all_other_agents_future_availability = np.zeros((other_agents_num, future_num_frames), dtype=np.float32)

    for idx, track_id in enumerate(list_agents_to_take):
        (
            agent_history_coords_offset,
            agent_history_yaws_offset,
            agent_history_extent,
            agent_history_availability,
        ) = get_relative_poses(
            history_num_frames_max + 1, history_frames, track_id, history_agents, agent_from_world, agent_yaw_rad
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
            future_num_frames, future_frames, track_id, future_agents, agent_from_world, agent_yaw_rad
        )
        all_other_agents_future_positions[idx] = agent_future_coords_offset
        all_other_agents_future_yaws[idx] = agent_future_yaws_offset
        all_other_agents_future_extents[idx] = agent_future_extent
        all_other_agents_future_availability[idx] = agent_future_availability

    # crop similar to ego above
    all_other_agents_history_positions[:, history_num_frames_agents + 1 :] *= 0
    all_other_agents_history_yaws[:, history_num_frames_agents + 1 :] *= 0
    all_other_agents_history_extents[:, history_num_frames_agents + 1 :] *= 0
    all_other_agents_history_availability[:, history_num_frames_agents + 1 :] *= 0

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

def map_func(lane_cfg_params, mapAPI, agent_centroid_m, agent_from_world, history_tl_faces):
      # START WORKING ON LANES
    # TODO (lberg): this implementation is super ugly, I'll fix it
    # TODO (anasrferreira): clean up more and add interpolation params to configuration as well.
    MAX_LANES = lane_cfg_params["max_num_lanes"]
    MAX_POINTS_LANES = lane_cfg_params["max_points_per_lane"]
    MAX_POINTS_CW = lane_cfg_params["max_points_per_crosswalk"]

    MAX_LANE_DISTANCE = lane_cfg_params["max_retrieval_distance_m"]
    INTERP_METHOD = InterpolationMethod.INTER_ENSURE_LEN  # split lane polyline by fixed number of points
    STEP_INTERPOLATION = MAX_POINTS_LANES  # number of points along lane
    MAX_CROSSWALKS = lane_cfg_params["max_num_crosswalks"]

    lanes_points = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES, 2), dtype=np.float32)
    lanes_availabilities = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES), dtype=np.float32)

    lanes_mid_points = np.zeros((MAX_LANES, MAX_POINTS_LANES, 2), dtype=np.float32)
    lanes_mid_availabilities = np.zeros((MAX_LANES, MAX_POINTS_LANES), dtype=np.float32)
    lanes_tl_feature = np.zeros((MAX_LANES, MAX_POINTS_LANES, 1), dtype=np.float32)

    # 8505 x 2 x 2
    lanes_bounds = mapAPI.bounds_info["lanes"]["bounds"]

    # filter first by bounds and then by distance, so that we always take the closest lanes
    lanes_indices = indices_in_bounds(agent_centroid_m, lanes_bounds, MAX_LANE_DISTANCE)
    distances = []
    for lane_idx in lanes_indices:
        lane_id = mapAPI.bounds_info["lanes"]["ids"][lane_idx]
        lane = mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)
        lane_dist = np.linalg.norm(lane["xyz_midlane"][:, :2] - agent_centroid_m, axis=-1)
        distances.append(np.min(lane_dist))
    lanes_indices = lanes_indices[np.argsort(distances)]

     # TODO: move below after traffic lights
    crosswalks_bounds = mapAPI.bounds_info["crosswalks"]["bounds"]
    crosswalks_indices = indices_in_bounds(agent_centroid_m, crosswalks_bounds, MAX_LANE_DISTANCE)
    crosswalks_points = np.zeros((MAX_CROSSWALKS, MAX_POINTS_CW, 2), dtype=np.float32)
    crosswalks_availabilities = np.zeros_like(crosswalks_points[..., 0])
    for i, xw_idx in enumerate(crosswalks_indices[:MAX_CROSSWALKS]):
        xw_id = mapAPI.bounds_info["crosswalks"]["ids"][xw_idx]
        points = mapAPI.get_crosswalk_coords(xw_id)["xyz"]
        points = transform_points(points[:MAX_POINTS_CW, :2], agent_from_world)
        n = len(points)
        crosswalks_points[i, :n] = points
        crosswalks_availabilities[i, :n] = True

    active_tl_faces = set(filter_tl_faces_by_status(history_tl_faces[0], "ACTIVE")["face_id"].tolist())
    active_tl_face_to_color: Dict[str, str] = {}
    for face in active_tl_faces:
        try:
            active_tl_face_to_color[face] = get_color_for_face(mapAPI, face)
        except KeyError:
            continue  # this happens only on KIRBY, 2 TLs have no match in the map

    for out_idx, lane_idx in enumerate(lanes_indices[:MAX_LANES]):
        lane_id = mapAPI.bounds_info["lanes"]["ids"][lane_idx]
        lane = mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)

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

        lanes_tl_feature[out_idx, :num_vectors_mid] = get_tl_feature_for_lane(mapAPI, lane_id, active_tl_face_to_color)

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



def vectorized_generate_agent_sample(
    state_index: int,
    frames: np.ndarray,
    agents: np.ndarray,
    tl_faces: np.ndarray,
    selected_track_id: Optional[int],
    render_context: RenderContext,
    history_num_frames_ego: int,
    history_num_frames_agents: int,
    future_num_frames: int,
    step_time: float,
    filter_agents_threshold: float,
    other_agents_num: int,
    max_agents_distance: float,
    mapAPI: MapAPI,
    lane_cfg_params: dict,
    rasterizer: Optional[Rasterizer] = None,
    perturbation: Optional[Perturbation] = None,
) -> dict:
    # TODO: doc string
    """ This will add to the generate_agent_sample function in l5kit to include all agents' history,
        needed for vectorized version.
    """
    history_num_frames_max = max(history_num_frames_ego, history_num_frames_agents)
    (
        history_frames,
        future_frames,
        history_agents,
        future_agents,
        history_tl_faces,
        future_tl_faces,
    ) = get_agent_context(state_index, frames, agents, tl_faces, history_num_frames_max, future_num_frames,)

    if perturbation is not None and len(future_frames) == future_num_frames:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        agent_type_idx = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        try:
            agent = filter_agents_by_track_id(
                filter_agents_by_labels(cur_agents, filter_agents_threshold), selected_track_id
            )[0]
        except IndexError:
            raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")
        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        agent_type_idx = np.argmax(agent["label_probabilities"])
        selected_agent = agent

    input_im = (
        None
        if not rasterizer
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    future_coords_offset, future_yaws_offset, future_extents, future_availability = get_relative_poses(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad
    )

    # For vectorized version we require both ego and agent history to be a Tensor of same length
    # => fetch history_num_frames_max for both, and later zero out frames exceeding the set history length.
    # Use history_num_frames_max + 1 because it also includes the current frame.
    history_coords_offset, history_yaws_offset, history_extents, history_availability = get_relative_poses(
        history_num_frames_max + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad
    )

    history_coords_offset[history_num_frames_ego + 1 :] *= 0
    history_yaws_offset[history_num_frames_ego + 1 :] *= 0
    history_extents[history_num_frames_ego + 1 :] *= 0
    history_availability[history_num_frames_ego + 1 :] *= 0

    history_vels_mps, future_vels_mps = compute_agent_velocity(history_coords_offset, future_coords_offset, step_time)

    d1 = {
        "image": input_im,
        "extent": agent_extent_m,
        "type": agent_type_idx,
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_world @ world_from_agent,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
            "target_positions": future_coords_offset,
        "target_yaws": future_yaws_offset,
        "target_extents": future_extents,
        "target_availabilities": future_availability.astype(np.bool),
        "history_positions": history_coords_offset,
        "history_yaws": history_yaws_offset,
        "history_extents": history_extents,
        "history_availabilities": history_availability.astype(np.bool),
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "speed": np.linalg.norm(future_vels_mps[0]),
    }

    agent_dict = agents_func(history_coords_offset, history_yaws_offset, history_availability, history_agents, agent_centroid_m, max_agents_distance, selected_track_id, other_agents_num, history_num_frames_max, future_num_frames, history_frames, agent_from_world, agent_yaw_rad, future_frames, future_agents, history_num_frames_agents)
    map_dict = map_func(lane_cfg_params, mapAPI, agent_centroid_m, agent_from_world, history_tl_faces)

    return {**d1, **agent_dict, **map_dict}

def filter_agents_by_distance(agents: np.ndarray, centroid: np.ndarray, max_distance: float) -> np.ndarray:
    """Filter agents by distance, cut to `max_distance` and sort the result
    Args:
        agents (np.ndarray): array of agents
        centroid (np.ndarray): centroid towards which compute distance
        max_distance (float): max distance to cut off
    Returns:
        np.ndarray: agents sorted and cut to max_distance
    """
    agents_dist = np.linalg.norm(agents["centroid"] - centroid, axis=-1)
    agents = agents[agents_dist < max_distance]
    agents_dist = agents_dist[agents_dist < max_distance]
    agents = agents[np.argsort(agents_dist)]
    return agents

def get_tl_feature_for_lane(map_api, lane_id, active_tl_face_to_color) -> int:
    """ Get traffic light feature for a lane given its active tl faces and a constant priority map.
    """
    # Map from traffic light state to its feature and priority index (to disambiguate multiple active tl faces)
    # "None" state is special and mean that a lane does not have a traffic light. "Unknown" means that the traffic
    # light exists but PCP cannot detect its state.
    # Except for `none`, priority increases with numbers
    tl_color_to_priority_idx = {"unknown": 0, "green": 1, "yellow": 2, "red": 3, "none": 4}

    lane_tces = map_api.get_lane_traffic_control_ids(lane_id)
    lane_tls = [tce for tce in lane_tces if map_api.is_traffic_light(tce)]
    if len(lane_tls) == 0:
        return tl_color_to_priority_idx["none"]

    active_tl_faces = active_tl_face_to_color.keys() & lane_tces
    # The active color with higher priority is kept
    highest_priority_idx_active = tl_color_to_priority_idx["unknown"]
    for active_tl_face in active_tl_faces:
        active_color = active_tl_face_to_color[active_tl_face]
        highest_priority_idx_active = max(highest_priority_idx_active, tl_color_to_priority_idx[active_color])
    return highest_priority_idx_active
    
@lru_cache(maxsize=CACHE_SIZE)
def get_color_for_face(mapAPI: MapAPI, face_id: str) -> str:
    """Obtain the color for the given face_id"""
    for color in ["red", "yellow", "green"]:
        if mapAPI.is_traffic_face_color(face_id, color):
            return color
    assert False, f"Face {face_id} has no valid color."
    return 0

# TODO @lberg: we're missing the AV
def get_other_agents_ids(
    all_agents_ids: np.ndarray, priority_ids: np.ndarray, selected_track_id: Optional[int], max_agents: int
) -> List[np.uint64]:
    """Get ids of agents around selected_track_id. Give precedence to `priority_ids`
    over `all_agents_ids` and cut to `max_agents`
    Args:
        all_agents_ids (np.ndarray): ids of all the agents from present to past
        priority_ids (np.ndarray): ids of agents we know are reliable in the present
        selected_track_id (Optional[int]): current id of the agent of interest
        max_agents (int): max agents to take
    Returns:
        List[np.uint64]: the list of track ids of agents to take
    """
    agents_taken: Set[np.uint64] = set()
    # ensure we give priority to reliable, then fill starting from the past
    for agent_id in np.concatenate([priority_ids, all_agents_ids]):
        if len(agents_taken) >= max_agents:
            break
        if agent_id != selected_track_id:
            agents_taken.add(agent_id)
    return list(agents_taken)


class EgoDatasetVectorized(EgoDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        mapAPI: MapAPI,
        perturbation: Optional[Perturbation] = None,
    ):
        super().__init__(cfg, zarr_dataset, rasterizer, perturbation)
        self.mapAPI = mapAPI

        # replace the sample function to access other agents
        render_context = RenderContext(
            raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
            set_origin_to_bottom=cfg["raster_params"]["set_origin_to_bottom"],
        )
        self.sample_function = partial(
            vectorized_generate_agent_sample,
            render_context=render_context,
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            step_time=cfg["model_params"]["step_time"],
            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
            other_agents_num=cfg["data_generation_params"]["other_agents_num"],
            max_agents_distance=cfg["data_generation_params"]["max_agents_distance"],
            mapAPI=mapAPI,
            lane_cfg_params=cfg["data_generation_params"]["lane_params"],
            rasterizer=rasterizer,
            perturbation=perturbation,
        )

    def get_scene_dataset(self, scene_index: int) -> "EgoDatasetVectorized":
        dataset = super().get_scene_dataset(scene_index).dataset
        return EgoDatasetVectorized(self.cfg, dataset, self.rasterizer, self.mapAPI, self.perturbation)