from abc import abstractmethod

import torch
from typing_extensions import Protocol

from l5kit.evaluation import error_functions
from l5kit.evaluation import metrics as l5metrics


# TODO(perone): mocking the output for now
class SimulationOutput:
    def __init__(self) -> None:
        self.simulated_ego_states = torch.ones(10)
        self.recorded_ego_states = torch.ones(10)


class SupportsMetricCompute(Protocol):
    """Protocol supporting the computation method for metrics."""
    metric_name: str

    @abstractmethod
    def compute(self, simulation_output: SimulationOutput) -> torch.Tensor:
        """The compute method sould return the result of the metric
        computed at every frame of the scene.

        :param simulation_output: the output from the closed-loop simulation
        :returns: a tensor with the result of the metric per frame
        """
        raise NotImplementedError


class DisplacementErrorMetric(SupportsMetricCompute):
    """Displacement error computes the elementwise distance from the
    simulated trajectory and the observed trajectory.

    :param error_function: error function to compute distance
    """
    metric_name = "displacement_error"

    def __init__(self, error_function: error_functions.ErrorFunction) -> None:
        self.error_function = error_function

    def compute(self, simulation_output: SimulationOutput) -> torch.Tensor:
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


class DistanceToRefTrajectoryMetric:
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

    def compute(self, simulation_output: SimulationOutput) -> torch.Tensor:
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
