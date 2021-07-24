from abc import ABC, abstractmethod
from typing import Any, DefaultDict, Dict, List

from l5kit.cle import composite_metrics, metrics, validators
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import SimulationOutputCLE, UnrollInputOutput


class SimulationConfigGym(SimulationConfig):
    """Defines the default parameters used for the simulation of ego and agents around it in L5Kit Gym.

    :param eps_length: the number of step to simulate per episode in the gym environment.
    """

    def __new__(cls, eps_length: int = 32) -> 'SimulationConfigGym':
        """Constructor method
        """
        self = super(SimulationConfigGym, cls).__new__(cls, use_ego_gt=False, use_agents_gt=True,
                                                       disable_new_agents=False, distance_th_far=30,
                                                       distance_th_close=15, num_simulation_steps=eps_length)
        return self


class SimulationOutputGym(SimulationOutputCLE):
    """This object holds information about the result of the simulation loop
    for a given scene dataset in gym-compatible L5Kit environment.

    :param scene_id: the scene indices
    :param sim_dataset: the simulation dataset
    :param ego_ins_outs: all inputs and outputs for ego (each frame of each scene has only one)
    :param agents_ins_outs: all inputs and outputs for agents (multiple per frame in a scene)
    """

    def __init__(self, scene_id: int, sim_dataset: SimulationDataset,
                 ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]],
                 agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]]):
        """Constructor method
        """
        super(SimulationOutputGym, self).__init__(scene_id, sim_dataset, ego_ins_outs, agents_ins_outs)

        # Required for Bokeh Visualizer
        simulated_dataset = sim_dataset.scene_dataset_batch[scene_id]
        self.tls_frames = simulated_dataset.dataset.tl_faces
        self.agents_th = simulated_dataset.cfg["raster_params"]["filter_agents_threshold"]


class CLEBaseMetricSet(ABC):
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


class L5MetricSet(CLEBaseMetricSet):
    """This class is responsible for computing a set of metric parametrization for the L5Kit.

    :param metric_prefix: this is a prefix that will identify the metric set being used.
    """

    def __init__(self, metric_prefix: str = "L5") -> None:
        """Constructor method
        """
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


class L5GymCLEMetricSet(L5MetricSet):
    """This class is responsible for computing a set of metric parametrization
    for the L5Kit Gym-compatible environment.

    :param metric_prefix: this is a prefix that will identify the metric set being used.
    """

    def __init__(self, metric_prefix: str = "L5_Gym_CLE") -> None:
        """Constructor method
        """
        super().__init__(metric_prefix)

    def build_metrics(self) -> List[metrics.SupportsMetricCompute]:
        """Returns a list of metrics that will be computed.
        """
        return [
            metrics.DisplacementErrorL2Metric(),
            metrics.YawErrorCAMetric()
        ]
