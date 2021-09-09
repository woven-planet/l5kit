
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
from l5kit.vectorization.vectorizer import Vectorizer


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
    vectorizer: Vectorizer,
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

    vec_dict = vectorizer.vectorize(selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world, history_frames, history_agents, history_tl_faces, history_coords_offset, history_yaws_offset, history_availability, future_frames, future_agents)

    return {**d1, **vec_dict}


class EgoDatasetVectorized(EgoDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        vectorizer: Vectorizer,
        perturbation: Optional[Perturbation] = None,
    ):
        super().__init__(cfg, zarr_dataset, rasterizer, perturbation)

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
            rasterizer=rasterizer,
            perturbation=perturbation,
            vectorizer=vectorizer
        )

    def get_scene_dataset(self, scene_index: int) -> "EgoDatasetVectorized":
        dataset = super().get_scene_dataset(scene_index).dataset
        return EgoDatasetVectorized(self.cfg, dataset, self.rasterizer, self.perturbation)