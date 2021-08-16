from typing import Dict, Iterable, List, Optional, Sequence

import torch

from l5kit.cle.composite_metrics import SupportsCompositeMetricCompute
from l5kit.cle.metrics import SupportsMetricCompute
from l5kit.cle.validators import SupportsMetricValidate, ValidatorOutput
from l5kit.simulation.unroll import SimulationOutputCLE


class EvaluationPlan:
    """Evaluation plan describes a plan to evaluate metrics and run
    validators. It is composed by the list of metrics that should be computed
    as well as the list of validators that will run. It checks for consistency
    of the plan (validators depend on metrics).

    .. note:: Please note that the intervention_validators argument, that
              specifies a list of interventions will stop the validation
              of all other metrics if any validator specified in this list
              is triggered.

    :param metrics: list of the metrics to compute
    :param validators: list of validators to compute
    :param composite_metrics: list of composite metrics to compute
    :param intervention_validators: list of validators that are considered
                                    interventions.
    """

    def __init__(self, metrics: Iterable[SupportsMetricCompute],
                 validators: Optional[Iterable[SupportsMetricValidate]] = None,
                 composite_metrics: Optional[Iterable[SupportsCompositeMetricCompute]] = None,
                 intervention_validators: Optional[List[str]] = None):
        self.metrics = metrics
        self.validators = validators or []
        self.composite_metrics = composite_metrics or []
        self.intervention_validators = intervention_validators or []
        self._validate_plan()

    def metrics_dict(self) -> Dict[str, SupportsMetricCompute]:
        """Get the metric names and metrics from the plan."""
        return {m.metric_name: m for m in self.metrics}

    def validators_dict(self) -> Dict[str, SupportsMetricValidate]:
        """Get the validator names and validators from the plan."""
        return {v.validator_name: v for v in self.validators}

    def composite_metrics_dict(self) -> Dict[str, SupportsCompositeMetricCompute]:
        """Get the composite metric names and composite metrics from the plan."""
        return {cm.composite_metric_name: cm for cm in self.composite_metrics}

    def _validate_plan(self) -> None:
        """Check for consistency of the plan, all metrics required by
        validators should be specified in the plan.
        """
        # Check for repetition of metrics and validators
        metric_names = [m.metric_name for m in self.metrics]
        validator_names = [v.validator_name for v in self.validators]
        composite_metric_names = [cm.composite_metric_name for cm in self.composite_metrics]

        if len(set(metric_names)) != len(metric_names):
            raise RuntimeError("You cannot have repeated metric names.")

        if len(set(validator_names)) != len(validator_names):
            raise RuntimeError("You cannot have repeated validator names.")

        if len(set(composite_metric_names)) != len(composite_metric_names):
            raise RuntimeError("You cannot have repeated composite metric names.")

        # Check if we have all validators specified in the intervention list
        for vname in self.intervention_validators:
            if vname not in validator_names:
                raise RuntimeError(f"Validator '{vname}' not found in validators list.")

        # Check for consistency of the validators, if we have all required
        # metrics especified.
        metric_calculators = self.metrics_dict()
        for validator in self.validators:
            for metric_requirement in validator.requires_metric:
                if metric_requirement not in metric_calculators:
                    raise RuntimeError(f"Metric '{metric_requirement}' required "
                                       f"by validator '{validator.validator_name}'.")

        # Check for consistency of the composite metrics, if we have all required
        # metrics and validators specified.
        validators_specified = self.validators_dict()
        for cm in self.composite_metrics:
            # Check for metric requirements
            for metric_requirement in cm.requires_metric:
                if metric_requirement not in metric_calculators:
                    raise RuntimeError(f"Metric '{metric_requirement}' required "
                                       f"by composite metric '{cm.composite_metric_name}'.")
            # Check for validator requirements
            for validator_requirement in cm.requires_validator:
                if validator_requirement not in validators_specified:
                    raise RuntimeError(f"Validator '{validator_requirement}' required "
                                       f"by composite metric '{cm.composite_metric_name}'.")

    def evaluate(self, simulation_output: SimulationOutputCLE) -> Dict[str, torch.Tensor]:
        """Execute the evaluation (metric computation) on the scene.

        :param simulation_output: output from the closed-loop simulator
        :returns: results from all metrics on a dictionary
        """
        results: Dict[str, torch.Tensor] = {}
        for metric_calculator in self.metrics:
            metric_result = metric_calculator.compute(simulation_output)
            results[metric_calculator.metric_name] = metric_result
        return results

    def evaluate_composite(self, simulation_output: SimulationOutputCLE,
                           scene_metrics: Dict[str, torch.Tensor],
                           scene_validation: Dict[str, ValidatorOutput]) -> Dict[str, float]:
        """Execute the evaluation of the composite metrics on the scene.

        :param simulation_output: output from the closed-loop simulator
        :param scene_metrics: metric results indexed by the metric name
        :param scene_validation: outputs from validator indexed by the validation name
        :return: results from the composite metrics indexed by the composite metric name
        """
        results: Dict[str, float] = {}
        for cm in self.composite_metrics:
            # Composite metrics should only see metrics and validators they require
            required_metrics = cm.requires_metric
            required_validators = cm.requires_validator
            # Filter to keep required metrics only
            scene_metrics_required = {
                metric_name: scene_metrics[metric_name]
                for metric_name in required_metrics
            }
            # Filter to keep required validators only
            scene_validators_required = {
                validator_name: scene_validation[validator_name]
                for validator_name in required_validators
            }

            cm_result = cm.compute(scene_metrics_required,
                                   scene_validators_required,
                                   simulation_output)
            results[cm.composite_metric_name] = cm_result
        return results

    def process_interventions(self,
                              results: Dict[str, ValidatorOutput]) -> Dict[str, ValidatorOutput]:
        """This method will process the the validator results accordingly
        to the validators defined as interventions. If any validator, that
        is also an intervention is triggered, it will reset all other validators.

        :param results: resuls from the validation
        :returns: updated the results
        """
        min_intervention_name = None
        min_intenvention_frame = float('inf')

        # For each intervention validator in the list
        for ivname in self.intervention_validators:
            voutput: ValidatorOutput = results[ivname]
            failed_frames: List[int] = voutput.failed_frames
            if len(failed_frames) <= 0:
                continue

            inner_min_failed_frame = min(failed_frames)
            if inner_min_failed_frame < min_intenvention_frame:
                min_intenvention_frame = inner_min_failed_frame
                min_intervention_name = ivname

        # If an intervention happened, all other validators are reset
        # as if they were not triggered.
        if min_intervention_name is not None:
            new_results = {ivname: ValidatorOutput(True, [])
                           for ivname in results.keys()}
            new_results[min_intervention_name] = results[min_intervention_name]
            return new_results
        else:
            return results

    def validate(self, scene_metrics: Dict[str, torch.Tensor],
                 simulation_output: SimulationOutputCLE) -> Dict[str, ValidatorOutput]:
        """Execute the validation (validators) on all metric results.

        :param scene_metrics: the result for the metrics computation
        :param simulation_output: output from the closed-loop simulator
        :returns: the result of all validators
        """
        results: Dict[str, ValidatorOutput] = {}
        for metric_validator in self.validators:
            # Validators should only see metrics they require
            required_metrics = metric_validator.requires_metric
            scene_metrics_required = {
                metric_name: scene_metrics[metric_name] for metric_name in required_metrics
            }
            validator_result = metric_validator.validate(scene_metrics_required,
                                                         simulation_output)
            results[metric_validator.validator_name] = validator_result

        # If we should stop at interventions, given a list of interventions
        if len(self.intervention_validators) > 0:
            results = self.process_interventions(results)

        return results


class ClosedLoopEvaluator:
    """The closed loop evaluator executes a evaluation plan and keep
    track of histograms, failed scenes, etc.

    :param evaluation_plan: the specified evaluation plan
    """

    #: Results of the metrics indexed by the scene id
    scene_metric_results: Dict[int, Dict[str, torch.Tensor]]
    #: Results from the validation results indexed by the scene id
    scene_validation_results: Dict[int, Dict[str, ValidatorOutput]]
    #: Results from the composite metrics indexed by the scene id
    scene_composite_metric_results: Dict[int, Dict[str, float]]

    def __init__(self, evaluation_plan: EvaluationPlan):
        self.evaluation_plan = evaluation_plan
        self.scene_validation_results = {}
        self.scene_metric_results = {}
        self.scene_composite_metric_results = {}

    def reset(self) -> None:
        """Resets the computed stats."""
        self.scene_validation_results = {}
        self.scene_metric_results = {}
        self.scene_composite_metric_results = {}

    def metric_results(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Return the computed metric results.

        :return: a dictionary indexed by scene with metric name
                 and results.
        """
        return self.scene_metric_results

    def validation_results(self) -> Dict[int, Dict[str, ValidatorOutput]]:
        """Return the computed validator results.

        :return: a dictionary indexed by scene with validator name
                 and results.
        """
        return self.scene_validation_results

    def composite_metric_results(self) -> Dict[int, Dict[str, float]]:
        """Return the computed composite metric results.

        :return: a dictionary indexed by scene with composite metric name
                 and results.
        """
        return self.scene_composite_metric_results

    def evaluate(self, simulation_outputs: Sequence[SimulationOutputCLE]) -> None:
        """Executes the evaluation plan on all outputs from the simulator.

        :param simulation_outputs: the outputs from the simulator
        """
        # TODO(perone): local parallelization can be done here
        for simulation_output in simulation_outputs:
            scene_id = simulation_output.get_scene_id()

            # Run metric calculation, metrics_for_scene here is a dict
            # where the tensor is the same size of the scene
            metrics_for_scene: Dict[str, torch.Tensor] = \
                self.evaluation_plan.evaluate(simulation_output)
            self.scene_metric_results[scene_id] = metrics_for_scene

            # Run validators
            validation_for_scene = self.evaluation_plan.validate(metrics_for_scene,
                                                                 simulation_output)
            self.scene_validation_results[scene_id] = validation_for_scene

            # Run composite metrics: these are metrics that depend on the result
            # of validators and metrics
            cm_for_scene = self.evaluation_plan.evaluate_composite(simulation_output,
                                                                   metrics_for_scene,
                                                                   validation_for_scene)
            self.scene_composite_metric_results[scene_id] = cm_for_scene
