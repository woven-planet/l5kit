import warnings
from typing import Callable, Tuple

import numpy as np
from transforms3d.euler import euler2mat, mat2euler

from ..geometry import rotation33_as_yaw, yaw_as_rotation33
from .ackerman_steering_model import fit_ackerman_model_exact
from .perturbation import Perturbation


#  if the offset or norm val is below this, we don't apply perturbation.
NUMERICAL_THRESHOLD = 1e-5


def _get_trajectory(
    history_frames: np.ndarray, future_frames: np.ndarray
) -> np.ndarray:
    num_history_frames = len(history_frames)
    num_future_frames = len(future_frames)
    total_num_frames = num_history_frames + num_future_frames

    trajectory = np.zeros((total_num_frames, 3), dtype=np.float32)

    # Note that history frames go backward in time from the anchor frame.
    trajectory[:num_history_frames, :2] = history_frames["ego_translation"][::-1, :2]
    trajectory[:num_history_frames, 2] = [
        rotation33_as_yaw(rot) for rot in history_frames["ego_rotation"][::-1]
    ]

    trajectory[num_history_frames:, :2] = future_frames["ego_translation"][:, :2]
    trajectory[num_history_frames:, 2] = [rotation33_as_yaw(rot) for rot in future_frames["ego_rotation"]]

    return trajectory


def _compute_speed(trajectory: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)


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

        lateral_offset_m, longitudinal_offset_m, yaw_offset_rad = self.random_offset_generator()
        position_offset_m = np.array([longitudinal_offset_m, lateral_offset_m, 0])

        num_history_frames = len(history_frames)
        num_future_frames = len(future_frames)
        total_num_frames = num_history_frames + num_future_frames
        if total_num_frames < 2:  # TODO is this an error?
            #  we need at least 2 frames to compute speed and steering rate.
            return history_frames.copy(), future_frames.copy()

        trajectory = _get_trajectory(history_frames, future_frames)
        speeds = _compute_speed(trajectory[:, :2])

        # TODO: ackerman perturbation does not work when EGO is static
        if np.sum(speeds) < NUMERICAL_THRESHOLD:
            return history_frames.copy(), future_frames.copy()

        curr_frame_idx = num_history_frames - 1
        position_offset_m = np.matmul(yaw_as_rotation33(trajectory[curr_frame_idx, 2]) , position_offset_m)

        #  perform ackerman steering model fitting
        gx = trajectory[curr_frame_idx + 1:, 0]
        gy = trajectory[curr_frame_idx + 1:, 1]
        gr = trajectory[curr_frame_idx + 1:, 2]
        gv = _compute_speed(trajectory[curr_frame_idx:, :2])

        x0 = trajectory[curr_frame_idx, 0] + position_offset_m[0]
        y0 = trajectory[curr_frame_idx, 1] + position_offset_m[1]
        r0 = trajectory[curr_frame_idx, 2] + yaw_offset_rad
        v0 = gv[0]

        wgx = np.ones(num_future_frames)
        wgx[-1] = 5
        wgy = np.ones(num_future_frames)
        wgy[-1] = 5
        wgr = np.zeros(num_future_frames)
        wgr[-1] = 5
        wgv = np.zeros(num_future_frames)

        new_xs, new_ys, new_yaws, new_vs, new_acc, new_steer = fit_ackerman_model_exact(
            x0, y0, r0, v0, gx, gy, gr, gv, wgx, wgy, wgr, wgv,
        )

        new_trajectory = np.stack((new_xs, new_ys, new_yaws), axis=1)

        history_frames["ego_translation"][0, 0] = x0
        history_frames["ego_translation"][0, 1] = y0
        history_frames["ego_rotation"][0] = yaw_as_rotation33(r0)

        future_frames["ego_translation"][:, :2] = new_trajectory[:, :2]
        future_frames["ego_rotation"] = np.array([yaw_as_rotation33(yaw) for yaw in new_trajectory[:, 2]])

        return history_frames, future_frames
