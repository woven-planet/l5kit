from abc import abstractmethod
from typing import Dict, List, Type

import torch
from typing_extensions import Protocol

from l5kit.cle import metrics, validators
from l5kit.cle.metrics import SupportsMetricCompute
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


class PassedDrivenMilesCompositeMetric(SupportsCompositeMetricCompute):
    """This composite metric will compute the "passed driven miles", which
    represents the sum of driven miles up to a first intervention frame
    that is detected based on the list of specified validators.

    :param intervention_validators: an intervention is a defined by any
                                    failed frame from any of the specified
                                    validators.
    :param driven_miles_metric: the metric that should be used as driven
                                miles (defaults to simulated driven miles),
                                but can also use the driven miles by the
                                log replay.
    :param ignore_entire_scene: if the entire driven miles from the scene
                                should be ignored when there was an intervention.
    """
    #: Name of the composite metric
    composite_metric_name: str
    #: List of names for metrics this composite metric depends on
    requires_metric: List[str]
    #: List of validators that this composite metric depends on
    requires_validator: List[str]

    def __init__(self, composite_metric_name: str,
                 intervention_validators: List[str],
                 driven_miles_metric: Type[SupportsMetricCompute] =
                 metrics.SimulatedDrivenMilesMetric,
                 ignore_entire_scene: bool = False):
        self.composite_metric_name = composite_metric_name
        self.requires_metric = [driven_miles_metric.metric_name]
        self.requires_validator = list(set(intervention_validators))
        self.ignore_entire_scene = ignore_entire_scene

    def compute(self, metric_results: Dict[str, torch.Tensor],
                validation_results: Dict[str, validators.ValidatorOutput],
                simulation_output: SimulationOutput) -> float:
        """Computes the driven miles until the first intervention
        is found (first failed frame).

        :param metric_results: results of the metrics required
        :tensor metric_results: [N], where N is the number of frames in the scene
        :param validation_results: results from the validators required
        :param simulation_output: output from the closed-loop simulator
        :returns: passed driven miles
        """
        driven_miles_result = metric_results[self.requires_metric[0]]

        min_all_frame_failed: List[int] = [driven_miles_result.size(0)]
        for validator_name in self.requires_validator:
            validator_results = validation_results[validator_name]
            if len(validator_results.failed_frames) > 0:
                # If ignore the entire scene is enabled, we just
                # return 0 passed miles
                if self.ignore_entire_scene:
                    return 0.0

                # ... otherwise, we check for the minimum failed
                # frame to compute the proportional driven miles
                min_frame_failed = min(validator_results.failed_frames)
                min_all_frame_failed.append(min_frame_failed)

        min_frame_failed = min(min_all_frame_failed)
        passed_driven_miles = driven_miles_result[:min_frame_failed].sum()
        passed_driven_miles_cpu = passed_driven_miles.cpu().item()
        return float(passed_driven_miles_cpu)


class DrivenMilesCompositeMetric(SupportsCompositeMetricCompute):
    """Composite metric to accumulate the total driven miles.

    :param composite_metric_name: name of the composite metric
    :param driven_miles_metric: the metric that should be used
                                to accumulate its values, defaults
                                to the simulated driven miles
    """
    #: Name of the composite metric
    composite_metric_name: str
    #: List of names for metrics this composite metric depends on
    requires_metric: List[str]
    #: List of validators that this composite metric depends on
    requires_validator: List[str]

    def __init__(self, composite_metric_name: str,
                 driven_miles_metric: Type[SupportsMetricCompute] =
                 metrics.SimulatedDrivenMilesMetric):
        self.composite_metric_name = composite_metric_name
        self.driven_miles_metric_name = driven_miles_metric.metric_name
        self.requires_metric = [self.driven_miles_metric_name]
        self.requires_validator = []

    def compute(self, metric_results: Dict[str, torch.Tensor],
                validation_results: Dict[str, validators.ValidatorOutput],
                simulation_output: SimulationOutput) -> float:
        driven_miles = metric_results[self.driven_miles_metric_name].sum()
        driven_miles_cpu = driven_miles.cpu().item()
        return float(driven_miles_cpu)
