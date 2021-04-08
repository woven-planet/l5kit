from abc import abstractmethod
from typing_extensions import Protocol

import torch

from l5kit.evaluation import error_functions


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
