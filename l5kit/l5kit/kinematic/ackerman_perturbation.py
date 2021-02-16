import warnings
from typing import Callable, Tuple

import numpy as np

from ..geometry import rotation33_as_yaw, yaw_as_rotation33
from .ackerman_steering_model import fit_ackerman_model_exact
from .perturbation import Perturbation


def _get_trajectory(history_frames: np.ndarray, future_frames: np.ndarray) -> np.ndarray:
    num_frames = len(history_frames) + len(future_frames)
    trajectory = np.zeros((num_frames, 3), dtype=np.float32)

    # Note that history frames go backward in time from the anchor frame.
    trajectory[:, :2] = np.concatenate(
        (history_frames["ego_translation"][::-1, :2], future_frames["ego_translation"][:, :2])
    )
    rotations = np.concatenate((history_frames["ego_rotation"][::-1], future_frames["ego_rotation"]))
    trajectory[:, 2] = [rotation33_as_yaw(rot) for rot in rotations]

    return trajectory


def _compute_displacements(trajectory: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)


class AckermanPerturbation(Perturbation):
    def __init__(
        self,
        random_offset_generator: Callable,
        perturb_prob: float,
        min_displacement: float,
    ):
        """
        Apply Ackerman to get a feasible trajectory with probability perturb_prob.

        Args:
            random_offset_generator (RandomGenerator): a callable that yields 2 values
            perturb_prob (float): probability between 0 and 1 of applying the perturbation
            min_displacement (float): minimum displacement required to apply lateral & yaw perturbation
        """
        self.perturb_prob = perturb_prob
        self.min_displacement = min_displacement
        self.random_offset_generator = random_offset_generator
        if perturb_prob == 0:
            warnings.warn(
                "Consider replacing this object with None if no perturbation is intended", RuntimeWarning, stacklevel=2
            )

    def perturb(
        self, history_frames: np.ndarray, future_frames: np.ndarray, **kwargs: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() >= self.perturb_prob:
            return history_frames.copy(), future_frames.copy(), (1.0, 0.0)

        (
            lateral_offset_m,
            longitudinal_offset_m,
            yaw_offset_rad,
            speed_multiplier,
            speed_offset,
        ) = self.random_offset_generator()
        speed_multiplier = max(0.2, min(speed_multiplier, 1.8))
        speed_offset = np.abs(speed_offset)

        num_history_frames = len(history_frames)
        num_future_frames = len(future_frames)

        assert num_history_frames > 0, f"Number of history frames: {num_history_frames}"
        assert num_future_frames > 0, f"Number of future frames: {num_future_frames}"

        trajectory = _get_trajectory(history_frames, future_frames)

        # curr_frame_idx = num_history_frames - 1
        curr_frame_idx = 0
        displacements = np.linalg.norm(np.diff(trajectory[curr_frame_idx:, :2], axis=0), axis=1)

        # TODO: ackerman lateral & yaw perturbation does not work when EGO slow moving
        if np.sum(displacements) < self.min_displacement:
            lateral_offset_m = 0
            yaw_offset_rad = 0

        position_offset_m = np.array([longitudinal_offset_m, lateral_offset_m, 0])
        position_offset_m = np.matmul(yaw_as_rotation33(trajectory[curr_frame_idx, 2]) , position_offset_m)

        #  perform ackerman steering model fitting
        gx = trajectory[curr_frame_idx + 1:, 0]
        gy = trajectory[curr_frame_idx + 1:, 1]
        gr = trajectory[curr_frame_idx + 1:, 2]
        gv = displacements

        x0 = trajectory[curr_frame_idx, 0] + position_offset_m[0]
        y0 = trajectory[curr_frame_idx, 1] + position_offset_m[1]
        r0 = trajectory[curr_frame_idx, 2] + yaw_offset_rad
        v0 = gv[0] * speed_multiplier + speed_offset

        wgx = np.ones(num_future_frames + num_history_frames - 1)
        wgy = np.ones(num_future_frames + num_history_frames - 1)
        wgr = np.ones(num_future_frames + num_history_frames - 1)
        wgv = np.zeros(num_future_frames + num_history_frames - 1)

        new_xs, new_ys, new_yaws, new_vs, new_acc, new_steer = fit_ackerman_model_exact(
            x0, y0, r0, v0, gx, gy, gr, gv, wgx, wgy, wgr, wgv,
        )

        history_frames["ego_translation"][::-1, 0] = np.append([x0], new_xs[:num_history_frames - 1])
        history_frames["ego_translation"][::-1, 1] = np.append([y0], new_ys[:num_history_frames - 1])
        history_frames["ego_rotation"][::-1] = np.array([yaw_as_rotation33(yaw) for yaw in np.append([r0], new_yaws[:num_history_frames - 1])])

        future_frames["ego_translation"][:, 0] = new_xs[num_history_frames - 1:]
        future_frames["ego_translation"][:, 1] = new_ys[num_history_frames - 1:]
        future_frames["ego_rotation"] = np.array([yaw_as_rotation33(yaw) for yaw in new_yaws[num_history_frames - 1:]])

        return history_frames, future_frames, (speed_multiplier, speed_offset)
