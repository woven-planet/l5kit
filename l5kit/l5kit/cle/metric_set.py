from abc import ABC, abstractmethod
from typing import Any, Dict, List

from l5kit.cle import composite_metrics, metrics, validators
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.simulation.unroll import SimulationOutputCLE


class BaseMetricSet(ABC):
    """Base class interface for the metric sets."""
    #: The prefix that will identify this metric
    metric_prefix: str
    evaluator: ClosedLoopEvaluator

    @abstractmethod
    def evaluate(self, sim_outputs: List[SimulationOutputCLE]) -> None:
        """Run the CLE (Closed-loop Evaluator) on simulated scenes.

        :param sim_outputs: outputs from the simulator.
        """
        raise NotImplementedError

    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Perform all required aggregations and returns a dictionary
        composed by all results."""
        raise NotImplementedError


class L5MetricSet(BaseMetricSet):
    """This class is responsible for computing a set of metric parametrization for the L5Kit.

    :param metric_prefix: this is a prefix that will identify the metric set being used.
    """

    def __init__(self, metric_prefix: str = "L5") -> None:
        self.metric_prefix = metric_prefix
        metric_list = self.build_metrics()
        validators_list = self.build_validators()
        composite_metric_list = self.build_composite_metrics()
        validators_intervention = self.get_validator_interventions()
        # Builds the evaluation plan and run the closed loop evaluator
        self.evaluation_plan = EvaluationPlan(metric_list, validators_list,
                                              composite_metric_list, validators_intervention)
        self.evaluator = ClosedLoopEvaluator(self.evaluation_plan)

    @abstractmethod
    def build_metrics(self) -> List[metrics.SupportsMetricCompute]:
        """Returns a list of metrics that will be computed.
        """
        raise NotImplementedError

    def build_validators(self) -> List[validators.SupportsMetricValidate]:
        """Returns a list of validators that will operate on the computed metrics.
        """
        return []

    def get_validator_interventions(self) -> List[str]:
        """Returns a list of validators that are considered an intervention.
        """
        return []

    def build_composite_metrics(self) -> List[composite_metrics.SupportsCompositeMetricCompute]:
        """Return a list of composite metrics that should be computed. Composite
        metrics are metrics that depend upon metrics and validator results.
        """
        return []

    def evaluate(self, sim_outputs: List[SimulationOutputCLE]) -> None:
        """Run the CLE (Closed-loop Evaluator) on simulated scenes.

        :param sim_outputs: outputs from the simulator.
        """
        self.evaluator.evaluate(sim_outputs)

    def reset(self) -> None:
        """Reset the current state of the CLE (Closed-loop Evaluator).
        """
        self.evaluator.reset()

    def get_results(self) -> Dict[str, Any]:
        """Perform all required aggregations and returns a dictionary composed by all results.
        """
        raise NotImplementedError

    def aggregate_failed_frames(self) -> Dict[str, Any]:
        """This method will aggregate the failed scenes and will return
        a dictionary indexed by the validator name associated with a list
        with FailedFrame items containing the scene_id and the frame index
        that triggered the validator."""
        # Do not aggregate and reduce if we don't have validators
        if not self.build_validators():
            return {}

        validation_results = self.evaluator.validation_results()
        val_scene_frames_agg = validators.ValidationFailedFramesAggregator()
        val_scene_frames_results = val_scene_frames_agg.aggregate(validation_results)
        return val_scene_frames_results
