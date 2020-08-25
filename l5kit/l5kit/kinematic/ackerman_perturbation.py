import warnings
from typing import Callable, Tuple

import numpy as np

from ..geometry import rotation33_as_yaw, yaw_as_rotation33
from .ackerman_steering_model import fit_ackerman_model_approximate
from .perturbation import Perturbation

#  if the offset or norm val is below this, we don't apply perturbation.
NUMERICAL_THRESHOLD = 0.00001


# TODO add docstrings for functions in the module
def get_lateral_offset_at_idx(
    input_translations: np.ndarray, perturbation_idx: int, offset_distance: float
) -> np.ndarray:
    len_trajectory = input_translations.shape[0]
    if len_trajectory <= perturbation_idx + 1:
        # we need at least 1 trajectory point after the perturbation index to compute lateral direction.
        return np.array([0.0, 0.0], dtype=np.float32)

    point_to_be_perturbed = input_translations[perturbation_idx, :]
    # we use this to find the local lateral direction
    next_point_in_trajectory = input_translations[perturbation_idx + 1, :]

    traj_dir_at_perturbation_point = next_point_in_trajectory - point_to_be_perturbed
    #  the minimum distance between two consecutive trajectory points to go forward with direction computing.
    consecutive_point_distance_threshold = 0.00001
    #  the trajectory in the perturbation point may be a stationary point, so there's no direction info.
    #  in this case, we just look at the start and end of the array to get the overall motion direction.
    #  if that fails, we just don't perturb.

    if np.linalg.norm(traj_dir_at_perturbation_point) < consecutive_point_distance_threshold:
        traj_dir_at_perturbation_point = input_translations[-1, :] - input_translations[0, :]
        if np.linalg.norm(traj_dir_at_perturbation_point) < consecutive_point_distance_threshold:
            traj_dir_at_perturbation_point = np.array([0.0, 0.0], dtype=np.float32)
        else:
            traj_dir_at_perturbation_point /= np.linalg.norm(traj_dir_at_perturbation_point)
    else:
        traj_dir_at_perturbation_point /= np.linalg.norm(traj_dir_at_perturbation_point)

    offset_dir = np.array([traj_dir_at_perturbation_point[1], -traj_dir_at_perturbation_point[0]])

    return offset_distance * offset_dir


def _get_history_and_future_frames_as_joint_trajectory(
    history_frames: np.ndarray, future_frames: np.ndarray
) -> np.ndarray:
    num_history_frames = len(history_frames)
    num_future_frames = len(future_frames)
    total_trajectory_length = num_history_frames + num_future_frames

    combined_trajectory = np.zeros((total_trajectory_length, 3), dtype=np.float32)

    # Note that history frames go backward in time from the anchor frame.
    combined_trajectory[:num_history_frames, :2] = history_frames["ego_translation"][::-1, :2]
    combined_trajectory[:num_history_frames, 2] = [
        rotation33_as_yaw(rot) for rot in history_frames["ego_rotation"][::-1]
    ]

    combined_trajectory[num_history_frames:, :2] = future_frames["ego_translation"][:, :2]
    combined_trajectory[num_history_frames:, 2] = [rotation33_as_yaw(rot) for rot in future_frames["ego_rotation"]]

    return combined_trajectory


def _compute_speeds_from_positions(trajectory: np.ndarray) -> np.ndarray:
    xs = trajectory[:, 0]
    ys = trajectory[:, 1]
    speeds = np.zeros(xs.shape)

    speeds[:-1] = np.sqrt((ys[1:] - ys[:-1]) ** 2 + (xs[1:] - xs[:-1]) ** 2)
    speeds[-1] = speeds[-2]
    return speeds


class AckermanPerturbation(Perturbation):
    def __init__(self, random_offset_generator: Callable, perturb_prob: float):
        """
        Apply Ackerman to get a feasible trajectory with probability perturb_prob.

        Args:
            random_offset_generator (RandomGenerator): a callable that yields 2 values
            perturb_prob (float): probability between 0 and 1 of applying the perturbation
        """
        self.perturb_prob = perturb_prob
        self.random_offset_generator = random_offset_generator
        if perturb_prob == 0:
            warnings.warn(
                "Consider replacing this object with None if no perturbation is intended", RuntimeWarning, stacklevel=2
            )

    def perturb(
        self, history_frames: np.ndarray, future_frames: np.ndarray, **kwargs: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() >= self.perturb_prob:
            return history_frames.copy(), future_frames.copy()

        lateral_offset_distance, yaw_offset_angle = self.random_offset_generator()

        if np.abs(lateral_offset_distance) < NUMERICAL_THRESHOLD:
            warnings.warn("ack not applied because of low lateral_distance", RuntimeWarning, stacklevel=2)
            return history_frames.copy(), future_frames.copy()

        num_history_frames = len(history_frames)
        num_future_frames = len(future_frames)
        total_trajectory_length = num_history_frames + num_future_frames
        if total_trajectory_length < 2:  # TODO is this an error?
            #  we need at least 2 frames to compute speed and steering rate.
            return history_frames.copy(), future_frames.copy()

        new_trajectory_to_be_smoothed = _get_history_and_future_frames_as_joint_trajectory(
            history_frames, future_frames
        )

        # laterally move the anchor frame
        lateral_translation_offset = get_lateral_offset_at_idx(
            new_trajectory_to_be_smoothed, num_history_frames - 1, lateral_offset_distance
        )

        new_trajectory_to_be_smoothed[num_history_frames - 1, :2] += lateral_translation_offset

        # laterally rotate the anchor frame
        new_trajectory_to_be_smoothed[num_history_frames - 1, 2] += yaw_offset_angle

        #  perform ackerman steering model fitting
        #  TODO(sms): Replace the call below to a cleaned up implementation

        gx = new_trajectory_to_be_smoothed[:, 0].reshape((-1,))
        gy = new_trajectory_to_be_smoothed[:, 1].reshape((-1,))
        gr = new_trajectory_to_be_smoothed[:, 2].reshape((-1,))
        gv = _compute_speeds_from_positions(new_trajectory_to_be_smoothed[:, :2]).reshape((-1,))

        wx = 5 * np.ones(total_trajectory_length)
        wy = 5 * np.ones(total_trajectory_length)
        wr = 5 * np.ones(total_trajectory_length)
        wv = 5 * np.ones(total_trajectory_length)
        wgx = np.zeros(total_trajectory_length)
        wgx[[0, num_history_frames - 1, -1]] = 5
        wgy = np.zeros(total_trajectory_length)
        wgy[[0, num_history_frames - 1, -1]] = 5
        wgr = np.zeros(total_trajectory_length)
        wgr[[0, num_history_frames - 1, -1]] = 5
        wgv = np.zeros(total_trajectory_length)

        new_xs, new_ys, new_yaws, new_vs = fit_ackerman_model_approximate(
            gx, gy, gr, gv, wx, wy, wr, wv, wgx, wgy, wgr, wgv
        )

        new_trajectory = np.array(list(zip(new_xs, new_ys, new_yaws)))

        new_yaws_as_rotations = np.array([yaw_as_rotation33(pos_yaw[2]) for pos_yaw in new_trajectory])
        new_history_frames = history_frames.copy()
        new_future_frames = future_frames.copy()

        new_history_frames["ego_translation"][::-1, :2] = new_trajectory[:num_history_frames, :2]
        new_history_frames["ego_rotation"][::-1] = new_yaws_as_rotations[:num_history_frames]

        new_future_frames["ego_translation"][:, :2] = new_trajectory[num_history_frames:, :2]
        new_future_frames["ego_rotation"] = new_yaws_as_rotations[num_history_frames:]

        return new_history_frames, new_future_frames
