from abc import abstractmethod
from collections import defaultdict
from enum import IntEnum
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Type

import torch
from typing_extensions import Protocol

from l5kit.cle import metrics
from l5kit.simulation.unroll import SimulationOutputCLE, TrajectoryStateIndices


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
                 simulation_output: SimulationOutputCLE) -> ValidatorOutput:
        """Apply the validator on the metric results.

        :param metric_results: results from all computed metrics
        :param simulation_output: output from the closed-loop simulator
        :returns: True if validator passed, False otherwise
        """
        raise NotImplementedError


class DurationMode(IntEnum):
    """For more information about duration mode, see RangeValidator docs."""
    TOTAL = 0
    CONTINUOUS = 1


class RangeValidator(SupportsMetricValidate):
    """Validates a metric based on specified range. The range validator will
    check: min_value < metric_value < max_value. It will also check if the
    metric violated a maximum duration (see duration_mode parameter notes).

    :param validator_name: name of the validator
    :param metric: metric class used to validate
    :param min_value: minimum value allowed (inclusive)
    :param max_value: maximum value allowed (inclusive)
    :param violation_duration_s: maximum allowed duration in seconds
                                 where the metric can be violated
    :param duration_mode: the duration mode can be "total" or "continuous". In
                          "total" mode, all violations are summed per scene, and
                          if they exceed the violation_duration_s parameter, then
                          it will not pass the validation. In "continuous" mode,
                          the violation_duration_s parameter must be exceded
                          on a continuous violation of the metric in time (without
                          a gap of non-violation). This parameter is ignored
                          if violation_duration_s is zero.
    """

    def __init__(self, validator_name: str, metric: Type[metrics.SupportsMetricCompute],
                 min_value: Optional[float] = None, max_value: Optional[float] = None,
                 violation_duration_s: float = 0.0, duration_mode: DurationMode = DurationMode.TOTAL):
        # Requires at least one value specification
        if min_value is None and max_value is None:
            raise ValueError("At least one parameter must be "
                             "specified: min_value or max_value.")

        if min_value is not None and max_value is not None:
            if min_value >= max_value:
                raise ValueError("Minimum value cannot be greater or equal"
                                 " to the maximum value.")

        self.validator_name = validator_name
        self.metric_name = metric.metric_name
        self.requires_metric = [self.metric_name]
        self.min_value = min_value
        self.max_value = max_value
        self.validation_duration_s = violation_duration_s
        self.duration_mode = duration_mode

    @staticmethod
    def cumsum_with_reset(timestamp_diff: torch.Tensor,
                          validation_mask: torch.Tensor) -> torch.Tensor:
        """Cumulative sum (cumsum) that takes into consideration a valid mask.
        If the valid mask is False on a timestamp, it will reset the accumulator.

        :param timestamp_diff: timestamps differentiated (1D array)
        :param validation_mask: a boolean mask with valid/invalid timestamps (1D array)
        :return: cumulative sum (1D array)
        """
        cumsum = torch.zeros_like(timestamp_diff)
        accumulator = 0.0
        for idx, (ts, vmask) in enumerate(zip(timestamp_diff, validation_mask)):
            if not vmask:
                accumulator = 0.0
            else:
                accumulator += ts.item()
            cumsum[idx] = accumulator
        return cumsum

    def validate(self, metric_results: Dict[str, torch.Tensor],
                 simulation_output: SimulationOutputCLE) -> ValidatorOutput:
        """Apply the validator on the results of the metric computation.

        :param metric_results: all metric results
        :param simulation_output: output from the closed-loop simulator
        :returns: True if validator passed, False otherwise
        """
        result = metric_results[self.metric_name]
        validation_mask = torch.zeros_like(result, dtype=torch.bool)

        if self.min_value is not None:
            validation_mask |= result < self.min_value

        if self.max_value is not None:
            validation_mask |= result > self.max_value

        # Immediate failure if there is a violation and the allowed
        # duration is zero
        if self.validation_duration_s <= 0.0:
            # Log the failed frames into the scene tracker
            failed_frame_indexes = torch.nonzero(validation_mask).squeeze(1)

            failed_frame_indexes = failed_frame_indexes.cpu().numpy().tolist()

            is_valid_scene = len(failed_frame_indexes) == 0
            return ValidatorOutput(is_valid_scene, failed_frame_indexes)

        # If duration is greater than zero, then we check
        # if there was a violation greater than the
        # allowed duration
        ego_states = simulation_output.simulated_ego_states
        timestamps = ego_states[:, TrajectoryStateIndices.TIME.value]

        # Diff of the timestamps
        pad = torch.as_tensor([0], device=timestamps.device)
        pad_ts = torch.cat((pad, timestamps))
        ts_diff = pad_ts[1:] - pad_ts[:-1]

        # Total mode: we sum all violation durations
        if self.duration_mode == DurationMode.TOTAL:
            # Build a cumulative sum (masked by the validation mask)
            ts_valid_cumsum = (ts_diff * validation_mask).cumsum(dim=0)
            ts_valid_cumsum = ts_valid_cumsum * validation_mask

        # Continuous mode: we check if any of the durations
        # violated the constraint
        if self.duration_mode == DurationMode.CONTINUOUS:
            # Cumulative sum here is computed with reset, if there is a valid
            # metric computation between two consecutive chunks of invalid
            # metrics, then this valid frame will reset the cumulative sum
            # of the timestamp diffs
            ts_valid_cumsum = RangeValidator.cumsum_with_reset(ts_diff, validation_mask)

        # Check which timestamps violated the duration
        ts_cumsum_violated = ts_valid_cumsum > self.validation_duration_s

        # Get the frame indexes and track them
        violation_indexes = torch.nonzero(ts_cumsum_violated).squeeze(1)
        violation_indexes = violation_indexes.cpu().numpy().tolist()

        is_valid_scene = len(violation_indexes) == 0
        return ValidatorOutput(is_valid_scene, violation_indexes)


class SupportsValidationAggregation(Protocol):
    """Protocol supporting the validation aggregation. This aggregator
    is responsible for aggregating results from the the validators and
    also doing a reduction step across multiple distributed nodes.
    """

    def aggregate(self, scene_validation_results:
                  Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:
        """This method will aggregate scenes locally and then will
        do the reduction step to aggregate data across distributed
        nodes.

        :param scene_validation_results: results from validator
                                         outputs per scene
        :returns: any result (it can be a composite object with scenes and frames
                  or just a float value) indexed by the validator name.
        """
        raise NotImplementedError


class ValidationCountingAggregator(SupportsValidationAggregation):
    """This aggregator will count (sum) the amount of invalid scenes or
    optionally the amount of failed frames on each scene.

    :param failed_frames: if True, it will count the number of frames
                          failed instead of scenes failed.
    """

    def __init__(self, failed_frames: bool = False):
        self.failed_frames = failed_frames

    def aggregate_scenes(self, scene_validation_results:
                         Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:
        """Aggregate the scenes locally on each node. This method will just
        sum the number of invalid scenes across all scenes in the node.

        :param scene_validation_results: results from validator
                                         outputs per scene
        :returns: a dictionary with the validation metric name as keys
                  and the sum of invalid scenes
        """
        aggregation: DefaultDict[str, int] = defaultdict(int)
        for _, validator_dict in scene_validation_results.items():
            for validator_name, validator_output in validator_dict.items():
                # Aggregate the number of failed frames in the scene
                if self.failed_frames:
                    aggregation[validator_name] += len(validator_output.failed_frames)
                else:  # or the number of scenes
                    aggregation[validator_name] += not validator_output.is_valid_scene
        aggregation_torch = {k: torch.as_tensor(v) for k, v in aggregation.items()}
        return aggregation_torch

    def aggregate(self, scene_validation_results:
                  Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:
        """This method will aggregate scenes locally.

        :param scene_validation_results: results from validator
                                         outputs per scene
        :returns: any result (it can be a composite object with scenes and frames
                  or just a float value) indexed by the validator name.
        """
        agg_scenes = self.aggregate_scenes(scene_validation_results)
        return agg_scenes


class FailedFrame(NamedTuple):
    """A named-tuple composed of the scene if and the
    frame index on that scene that caused the validator to fail."""
    scene_id: int
    frame_index: int


class ValidationFailedFramesAggregator:
    """This aggregator will aggregate all failed frames (and scenes)."""

    def aggregate_scenes(self, scene_validation_results:
                         Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:
        """This method will aggregate the failed scene/frame tuples locally
        at each node and then build a 1D torch tensor with the unpacked tuples such
        as [scene, frame, scene frame, (...)], to be able to gather them later on
        the reduction across different nodes.

        :param scene_validation_results: results from validator
                                         outputs per scene
        :returns: a dictionary, indexed by the validator name, with
                  FailedFrame list containing the scene/frames failed.
        """
        aggregation: DefaultDict[str, List[FailedFrame]] = defaultdict(list)

        for scene_id, validator_dict in scene_validation_results.items():
            for validator_name, validator_output in validator_dict.items():
                if len(validator_output.failed_frames) > 0:
                    failed_fames = [FailedFrame(scene_id, frame_index)
                                    for frame_index in validator_output.failed_frames]
                    aggregation[validator_name].extend(failed_fames)

        aggregation_torch = {k: torch.as_tensor(v) for k, v in aggregation.items()}
        return aggregation_torch

    def aggregate(self, scene_validation_results:
                  Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:
        """This method will aggregate scenes locally and then will
        do the reduction step to aggregate data across distributed
        nodes.

        :param scene_validation_results: results from validator
                                         outputs per scene
        :returns: any result (it can be a composite object with scenes and frames
                  or just a float value) indexed by the validator name.
        """
        agg_scenes = self.aggregate_scenes(scene_validation_results)
        return agg_scenes
