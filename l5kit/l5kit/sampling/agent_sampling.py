from typing import Optional, Tuple

import numpy as np

from ..data import filter_agents_by_frame, filter_agents_by_labels, get_agent_by_track_id
from ..geometry import rotation33_as_yaw, world_to_image_pixels_matrix
from ..kinematic import Perturbation
from ..rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer
from .slicing import get_future_slice, get_history_slice


def generate_agent_sample(
    state_index: int,
    frames: np.ndarray,
    all_agents: np.ndarray,
    selected_track_id: Optional[int],
    raster_size: Tuple[int, int],
    pixel_size: np.ndarray,
    ego_center: np.ndarray,
    history_num_frames: int,
    history_step_size: int,
    future_num_frames: int,
    future_step_size: int,
    filter_agents_threshold: float,
    rasterizer: Optional[Rasterizer] = None,
    perturbation: Optional[Perturbation] = None,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Arguments:
        state_index {int} -- The anchor frame index, i.e. the "current" timestep
        frames {np.ndarray} -- The whole frames array, can be numpy array or a zarr array.
        all_agents {np.ndarray} -- The whole agents array, can be numpy array or a zarr array.
        selected_track_id: {Optional[int]} -- Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        raster_size {Tuple[int, int]} -- Desired output raster dimensions
        pixel_size {np.ndarray} -- Size of one pixel in the real world
        ego_center {np.ndarray} -- Where in the raster to draw the ego, [0.5,0.5] would be the center
        history_num_states {int} -- Amount of history frames to draw into the rasters.,
        history_step_size {int} -- Steps to take between frames, can be used to subsample history frames.
        future_num_states {int} -- Amount of history frames to draw into the rasters.
        future_step_size {int} -- Steps to take between targets into the future.
        filter_agents_threshold {float} -- Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent.
        rasterizer {Rasterizer} -- Rasterizer of some sort that draws a map image.

        voxel_shape (Tuple[int, int]): Desired output raster dimensions
        voxel_size (np.ndarray): Size of one pixel in the real world
        voxel_ego_center (np.ndarray): Where in the raster to draw the ego, [0.5,0.5] would be the center
        history_num_states (int): Amount of history frames to draw into the rasters.,
        history_step_size (int): Steps to take between frames, can be used to subsample history frames.
        future_num_states (int): Amount of history frames to draw into the rasters.
        future_step_size (int): Steps to take between targets into the future.
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
based on their probability of being a relevant agent.

    Keyword Arguments:
        rasterizer (Optional[Rasterizer]): Rasterizer of some sort that draws a map image.
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
to train models that can recover from slight divergence from training set data (default: {None})

    Raises:
        ValueError: A ValueError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        input_im (np.ndarray): Input raster to be used as input of a learned prediction or planning model.
        future_coords_offset (np.ndarray): The offset from the current state in terms of translation, currently
expressed in pixels (to be changed).

        future_yaws_offset (np.ndarray): The yaw offset of future frames from the current frame.
        future_availability (np.ndarray): A binary mask of ``future_num_states`` length whether there is a valid
        target for that state. If you sample near the end of a scene, this may contain zeroes.

    """
    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = get_history_slice(state_index, history_num_frames, history_step_size, include_current_state=True)
    history_frames = frames[history_slice]

    future_slice = get_future_slice(state_index, future_num_frames, future_step_size)
    future_frames = frames[future_slice]

    if perturbation is not None:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]

    if selected_track_id is None:
        agent_centroid = cur_frame["ego_translation"][:2]
        agent_yaw = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_agent = None
    else:
        # we must ensure the requested track is in the cur frame
        # otherwise we can not center it in the frame
        agent = get_agent_by_track_id(all_agents, cur_frame, selected_track_id)
        if agent is None:
            raise ValueError(f" track_id {selected_track_id} not in frame")
        if agent not in filter_agents_by_labels(
            filter_agents_by_frame(all_agents, cur_frame), filter_agents_threshold
        ):
            raise ValueError(f" track_id {selected_track_id} is in frame but under th {filter_agents_threshold}")
        agent_centroid = agent["centroid"]
        agent_yaw = float(agent["yaw"])
        agent_extent = agent["extent"]
        selected_agent = agent

    input_im = None if not rasterizer else rasterizer.rasterize(history_frames, all_agents, selected_agent)

    world_to_image_space = world_to_image_pixels_matrix(
        raster_size,
        pixel_size,
        ego_translation_m=agent_centroid,
        ego_yaw_rad=agent_yaw,
        ego_center_in_image_ratio=ego_center,
    )

    future_coords_offset, future_yaws_offset, future_availability = _create_targets_for_deep_prediction(
        future_num_frames, future_frames, selected_track_id, all_agents, agent_centroid[:2], agent_yaw,
    )

    return {
        "image": input_im,
        "target_positions": future_coords_offset,
        "target_yaws": future_yaws_offset,
        "target_availabilities": future_availability,
        "world_to_image": world_to_image_space,
        "centroid": agent_centroid,
        "yaw": agent_yaw,
        "extent": agent_extent,
    }


def _create_targets_for_deep_prediction(
    future_num_frames: int,
    future_frames: np.ndarray,
    selected_track_id: Optional[int],
    all_agents: np.ndarray,
    agent_centroid: np.ndarray,
    agent_yaw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Internal function that creates the targets and availability masks for deep prediction-type models.
    """
    # TODO add docstring
    # How much the future coordinates differ from the current state in meters.
    # This is generally used as the target in training a deep prediction model.
    future_coords_offset = np.zeros((future_num_frames, 2), dtype=np.float32)
    future_yaws_offset = np.zeros((future_num_frames, 1), dtype=np.float32)

    # 1 if a target is present, 0 if not. This can be used to multiply the loss.
    future_availability = np.zeros((future_num_frames, 3), dtype=np.float32)

    for i, frame in enumerate(future_frames):
        if selected_track_id is None:
            future_agent_centroid = frame["ego_translation"][:2]
            future_agent_yaw = rotation33_as_yaw(frame["ego_rotation"])
        else:
            # it's not guaranteed the target will be in every future frame
            future_agent = get_agent_by_track_id(all_agents, frame, selected_track_id)
            if future_agent is None:
                future_availability[i] = 0.0  # keep track of invalid futures
                continue
            future_agent_centroid = future_agent["centroid"]
            future_agent_yaw = future_agent["yaw"]

        future_coords_offset[i] = future_agent_centroid - agent_centroid
        future_yaws_offset[i] = future_agent_yaw - agent_yaw
        future_availability[i] = 1.0
    return future_coords_offset, future_yaws_offset, future_availability
