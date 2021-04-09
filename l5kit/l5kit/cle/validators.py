from abc import abstractmethod
from typing import Dict, List, NamedTuple

import torch
from typing_extensions import Protocol

from l5kit.simulation.unroll import SimulationOutput


class ValidatorOutput(NamedTuple):
    """Output from validators. Validators should return a boolean
    telling if the scene is valid or not and a list of failed
    frames."""
    is_valid_scene: bool
    failed_frames: List[int]


class SupportsMetricValidate(Protocol):
    """Protocol supporting the validation for metrics. The evaluation plan
    has two main components: metrics and validators. Metrics are completely
    independent, but validators not, as they depend on metrics, therefore
    the validator needs to carry a list of metrics it requires to compute,
    otherwise, the evaluation plan is not consistent, and this is checked
    by the evaluation plan.
    """
    validator_name: str
    requires_metric: List[str]

    @abstractmethod
    def validate(self, metric_results: Dict[str, torch.Tensor],
                 simulation_output: SimulationOutput) -> ValidatorOutput:
        """Apply the validator on the metric results.

        :param metric_results: results from all computed metrics
        :param simulation_output: output from the closed-loop simulator
        :returns: True if validator passed, False otherwise
        """
        raise NotImplementedError
