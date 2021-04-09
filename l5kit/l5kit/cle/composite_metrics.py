from abc import abstractmethod
from typing import Dict, List, Protocol

import torch

from l5kit.cle import validators
from l5kit.simulation.unroll import SimulationOutput


class SupportsCompositeMetricCompute(Protocol):
    """Protocol supporting the computation method for metrics."""

    #: Name of the composite metric
    composite_metric_name: str
    #: List of names for metrics this composite metric depends on
    requires_metric: List[str]
    #: List of validators that this composite metric depends on
    requires_validator: List[str]

    @abstractmethod
    def compute(self, metric_results: Dict[str, torch.Tensor],
                validation_results: Dict[str, validators.ValidatorOutput],
                simulation_output: SimulationOutput) -> float:
        """Method that supports the computation of the composite metric. This
        metric should return a single float per scene.

        :param metric_results: results of the metrics required
        :tensor metric_results: [N], where N is the number of frames in the scene
        :param validation_results: results from the validators required
        :param simulation_output: output from the closed-loop simulator
        :returns: a float result
        """
        raise NotImplementedError
