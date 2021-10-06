from abc import ABC, abstractmethod

import numpy as np
import torch
from typing_extensions import Protocol

from l5kit.data import get_agents_slice_from_frames
from l5kit.evaluation import error_functions
from l5kit.evaluation import metrics as l5metrics
from l5kit.rasterization import EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH
from l5kit.simulation.unroll import SimulationOutputCLE, TrajectoryStateIndices


class SupportsMetricCompute(Protocol):
    """Protocol supporting the computation method for metrics."""
    metric_name: str

    @abstractmethod
    def compute(self, simulation_output: SimulationOutputCLE) -> torch.Tensor:
        """The compute method sould return the result of the metric
        computed at every frame of the scene.

        :param simulation_output: the output from the closed-loop simulation
        :returns: a tensor with the result of the metric per frame
        """
        raise NotImplementedError


class CollisionMetricBase(ABC):
    """This is the abstract base class for the collision metric.

    :param collision_type: the type of collision to compute
    """
    @abstractmethod
    def __init__(self, collision_type: l5metrics.CollisionType) -> None:
        self.collision_type = collision_type

    def _compute_frame(self, simulated_agent_frame: np.ndarray,
                       simulated_frame_ego_state: torch.Tensor) -> float:
        """Detects collision per frame of the scene.

        :param observed_frame: the ground-truth frame
        :param simulated_frame_ego_state: ego state from the simulated frame,
                                          this is a 1D array with the frame
                                          ego state
        :returns: metric result for the frame, where 1 means a collision,
                  and 0 otherwise.
        """
        simulated_centroid = simulated_frame_ego_state[:TrajectoryStateIndices.Y + 1].cpu().numpy()
        simulated_angle = simulated_frame_ego_state[TrajectoryStateIndices.THETA].cpu().numpy()
        simulated_extent = np.r_[EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH]
        collision_ret = l5metrics.detect_collision(simulated_centroid, simulated_angle,
                                                   simulated_extent, simulated_agent_frame)
        if collision_ret is not None:
            if collision_ret[0] == self.collision_type:
                return 1.0

        return 0.0

    def compute(self, simulation_output: SimulationOutputCLE) -> torch.Tensor:
        """Compute the metric on all frames of the scene.

        :param simulation_output: the output from the closed-loop simulation
        :returns: collision per frame (a 1D array with the same size of the
                  number of frames, where 1 means a colision, 0 otherwise)
        """
        simulated_scene_ego_state = simulation_output.simulated_ego_states
        simulated_agents = simulation_output.simulated_agents
        simulated_egos = simulation_output.simulated_ego

        if len(simulated_agents) < len(simulated_scene_ego_state):
            raise ValueError("More simulated timesteps than observed.")

        num_frames = simulated_scene_ego_state.size(0)
        metric_results = torch.zeros(num_frames, device=simulated_scene_ego_state.device)
        for frame_idx in range(num_frames):
            simulated_ego_state_frame = simulated_scene_ego_state[frame_idx]
            simulated_ego_frame = simulated_egos[frame_idx]
            simulated_agent_frame = simulated_agents[get_agents_slice_from_frames(simulated_ego_frame)]
            result = self._compute_frame(simulated_agent_frame, simulated_ego_state_frame)
            metric_results[frame_idx] = result
        return metric_results


class CollisionFrontMetric(CollisionMetricBase):
    """Computes the front collision metric."""
    metric_name = "collision_front"

    def __init__(self) -> None:
        super().__init__(l5metrics.CollisionType.FRONT)


class CollisionRearMetric(CollisionMetricBase):
    """Computes the rear collision metric."""
    metric_name = "collision_rear"

    def __init__(self) -> None:
        super().__init__(l5metrics.CollisionType.REAR)


class CollisionSideMetric(CollisionMetricBase):
    """Computes the side collision metric."""
    metric_name = "collision_side"

    def __init__(self) -> None:
        super().__init__(l5metrics.CollisionType.SIDE)


class DisplacementErrorMetric(SupportsMetricCompute):
    """Displacement error computes the elementwise distance from the
    simulated trajectory and the observed trajectory.

    :param error_function: error function to compute distance
    """
    metric_name = "displacement_error"

    def __init__(self, error_function: error_functions.ErrorFunction) -> None:
        self.error_function = error_function

    def compute(self, simulation_output: SimulationOutputCLE) -> torch.Tensor:
        """Compute the metric on all frames of the scene.

        :param simulation_output: the output from the closed-loop simulation
        :returns: distance per frame [Shape: N, where N = timesteps]
        """
        simulated_scene_ego_state = simulation_output.simulated_ego_states
        simulated_centroid = simulated_scene_ego_state[:, :2]  # [Timesteps, 2]
        observed_ego_states = simulation_output.recorded_ego_states[:, :2]  # [Timesteps, 2]

        if len(observed_ego_states) < len(simulated_centroid):
            raise ValueError("More simulated timesteps than observed.")

        # Don't have simulation for all steps, have to clip it
        observed_ego_states_fraction = observed_ego_states[:len(simulated_centroid)]

        error = self.error_function(simulated_centroid, observed_ego_states_fraction)
        return error


class DisplacementErrorL2Metric(DisplacementErrorMetric):
    """Displacement error calculated with euclidean distance."""
    metric_name = "displacement_error_l2"

    def __init__(self) -> None:
        super().__init__(error_functions.l2_error)


class DistanceToRefTrajectoryMetric(SupportsMetricCompute):
    """Distance to reference trajectory metric. This metric will compute
    the distance from the predicted centroid to the closest waypoint
    in the reference trajectory.

    .. note::  Please note that this metric is different than the displacement
               error because it is taking into consideration the entire
               reference trajectory at each point of the simulated trajectory.

    :param scene_fraction: fraction of the simulated scene used to
                           evaluate against the reference trajectory. This
                           fraction should be between 0.0 and 1.0.
    """
    metric_name = "distance_to_reference_trajectory"

    def __init__(self, scene_fraction: float = 0.8) -> None:
        # Check fraction range
        if scene_fraction < 0.0 or scene_fraction > 1.0:
            raise ValueError("'screne_fraction' should be between 0.0 and 1.0.")

        self.scene_fraction = scene_fraction

    def compute(self, simulation_output: SimulationOutputCLE) -> torch.Tensor:
        """Compute the metric on all frames of the scene.

        :param simulation_output: the output from the closed-loop simulation
        :returns: distance to reference trajectory per
                  frame [Shape: N, where N = timesteps]
        """
        # Shape = [Timesteps, 7]
        simulated_scene_ego_state = simulation_output.simulated_ego_states
        simulated_centroid = simulated_scene_ego_state[:, :2]  # [Timesteps, 2]
        observed_ego_states = simulation_output.recorded_ego_states[:, :2]

        if len(observed_ego_states) < len(simulated_centroid):
            raise ValueError("More simulated timesteps than observed.")

        # Trim the simulated trajectory to have a specified fraction
        simulated_fraction_length = int(len(simulated_centroid) * self.scene_fraction)
        simulated_centroid_fraction = simulated_centroid[0:simulated_fraction_length]

        observed_ego_states = observed_ego_states.unsqueeze(0)
        distance = l5metrics.distance_to_reference_trajectory(simulated_centroid_fraction,
                                                              observed_ego_states)
        return distance


class SimulatedDrivenMilesMetric:
    """This metric will compute the driven miles per frame for the simulated
    trajectory (as opposed to the one in the log replay)."""
    metric_name = "simulated_driven_miles"
    METER_TO_MILES = 0.000621371

    def compute(self, simulation_output: SimulationOutputCLE) -> torch.Tensor:
        """Compute the metric on all frames of the scene.

        :param simulation_output: the output from the closed-loop simulation
        :returns: driven miles per each frame
        """
        simulated_scene_ego_state = simulation_output.simulated_ego_states
        simulated_centroid = simulated_scene_ego_state[:, :2]  # [Timesteps, 2]
        simulated_centroid = simulated_centroid.to(torch.float64)

        drive_meters = torch.linalg.norm(simulated_centroid[1:] - simulated_centroid[0:-1], dim=1)
        pad = torch.as_tensor([0.], device=simulated_centroid.device)
        pad_drive_meters = torch.cat((pad, drive_meters))
        driven_miles = pad_drive_meters * self.METER_TO_MILES
        return driven_miles


class ReplayDrivenMilesMetric:
    """This metric will compute the driven miles per frame for the observed
    trajectory, the one in the log replay (as opposed to the one simulated)."""
    metric_name = "replay_driven_miles"
    METER_TO_MILES = 0.000621371

    def compute(self, simulation_output: SimulationOutputCLE) -> torch.Tensor:
        """Compute the metric on all frames of the scene.

        :param simulation_output: the output from the closed-loop simulation
        :returns: driven miles per each frame
        """
        observed_ego_states_centroid = simulation_output.recorded_ego_states[:, :2]  # [Timesteps, 2]
        observed_ego_states_centroid = observed_ego_states_centroid.to(torch.float64)

        drive_meters = \
            torch.linalg.norm(observed_ego_states_centroid[1:]
                              - observed_ego_states_centroid[0:-1], dim=1)
        pad = torch.as_tensor([0.], device=observed_ego_states_centroid.device)
        pad_drive_meters = torch.cat((pad, drive_meters))
        driven_miles = pad_drive_meters * self.METER_TO_MILES
        return driven_miles


class YawErrorMetric(SupportsMetricCompute):
    """Yaw error computes the difference between the
    simulated trajectory yaw and the observed trajectory yaw.

    :param error_function: error function to compute distance
    """
    metric_name = "yaw_error"

    def __init__(self, error_function: error_functions.ErrorFunction = error_functions.closest_angle_error) -> None:
        self.error_function = error_function

    def compute(self, simulation_output: SimulationOutputCLE) -> torch.Tensor:
        """Compute the metric on all frames of the scene.

        :param simulation_output: the output from the closed-loop simulation
        :returns: distance per frame [Shape: N, where N = timesteps]
        """
        simulated_scene_ego_state = simulation_output.simulated_ego_states
        simulated_yaws = simulated_scene_ego_state[:, 2:3]  # [Timesteps,]
        observed_ego_yaws = simulation_output.recorded_ego_states[:, 2:3]  # [Timesteps,]

        if len(observed_ego_yaws) < len(simulated_yaws):
            raise ValueError("More simulated timesteps than observed.")

        # Don't have simulation for all steps, have to clip it
        observed_ego_yaws_fraction = observed_ego_yaws[:len(simulated_yaws)]

        error = self.error_function(simulated_yaws, observed_ego_yaws_fraction)
        return error


class YawErrorCAMetric(YawErrorMetric):
    """Yaw error calculated with closest angle."""
    metric_name = "yaw_error_closest_angle"

    def __init__(self) -> None:
        super().__init__(error_functions.closest_angle_error)


class SimulatedVsRecordedEgoSpeedMetric(SupportsMetricCompute):
    """This metric computes the speed delta between recorded and simulated ego.
    When simulated ego is traveling faster than recorded ego, this metric is > 0.
    When simulated ego is traveling slower than recorded ego, this metric is < 0.
    We can use this metric in conjunction with a RangeValidator to identify cases
    where simulated ego is consistently traveling much faster (or much slower) than recorded ego.
    """
    metric_name = "simulated_minus_recorded_ego_speed"

    def compute(self, simulation_output: SimulationOutputCLE) -> torch.Tensor:
        # calculate the speed error between simulated and recorded ego over the course of the simulation.
        simulated_centroid = simulation_output.simulated_ego_states[:, :2]
        simulated_velocity = simulated_centroid[1:] - simulated_centroid[:-1]
        simulated_speed = torch.linalg.norm(simulated_velocity, dim=1)

        recorded_centroid = simulation_output.recorded_ego_states[:, :2]
        recorded_velocity = recorded_centroid[1:] - recorded_centroid[:-1]
        recorded_speed = torch.linalg.norm(recorded_velocity, dim=1)

        # TODO: Replace 10 by (1 / step_time)
        return (simulated_speed - recorded_speed) * 10
