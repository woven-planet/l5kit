from typing import List

from l5kit.cle import composite_metrics as cm
from l5kit.cle import metric_set, metrics, validators


class L2DisplacementYawMetricSet(metric_set.L5MetricSet):
    """This class is responsible for computing a set of metric parametrization
    for the L5Kit Gym-compatible environment. The metrics in this class comprise
    L2 displacement error for the (x, y) coordinates and closest-angle error for the yaw.

    :param metric_prefix: this is a prefix that will identify the metric set being used.
    """

    def __init__(self, metric_prefix: str = "L2DisplacementYaw") -> None:
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


class CLEMetricSet(metric_set.L5MetricSet):
    """This class is responsible for computing a set of metric parametrization
    for the L5Kit Gym-compatible environment. The metrics in this class comprise
    the closed loop evaluation metrics of L5Kit.

    :param metric_prefix: this is a prefix that will identify the metric set being used.
    """

    def __init__(self, metric_prefix: str = "CLE") -> None:
        """Constructor method
        """
        super().__init__(metric_prefix)

    def build_metrics(self) -> List[metrics.SupportsMetricCompute]:
        """Returns a list of metrics that will be computed.
        """
        return [
            metrics.DisplacementErrorL2Metric(),
            metrics.DistanceToRefTrajectoryMetric(),
            metrics.CollisionFrontMetric(),
            metrics.CollisionRearMetric(),
            metrics.CollisionSideMetric(),
            metrics.SimulatedVsRecordedEgoSpeedMetric(),
            metrics.SimulatedDrivenMilesMetric(),
            metrics.ReplayDrivenMilesMetric(),
        ]

    def build_validators(self) -> List[validators.SupportsMetricValidate]:
        """Returns a list of validators that will operate on the computed metrics.
        """
        return [
            validators.RangeValidator("displacement_error_l2", metrics.DisplacementErrorL2Metric, max_value=30),
            validators.RangeValidator("distance_ref_trajectory", metrics.DistanceToRefTrajectoryMetric, max_value=4),
            # Collision indicator
            validators.RangeValidator("collision_front", metrics.CollisionFrontMetric, max_value=0),
            validators.RangeValidator("collision_rear", metrics.CollisionRearMetric, max_value=0),
            validators.RangeValidator("collision_side", metrics.CollisionSideMetric, max_value=0),
            # Passiveness indicator - slow_driving metric - Failure if simulated ego is slower than recording by more
            # than 5 m/s (~11 MPH) for 2.3 seconds
            validators.RangeValidator("passive_ego", metrics.SimulatedVsRecordedEgoSpeedMetric,
                                      min_value=-5.0, violation_duration_s=2.3,
                                      duration_mode=validators.DurationMode.CONTINUOUS),
            # Aggressiveness metrics - Failure if simulated ego is faster than recording by more
            # than 5 m/s (~11 MPH) for 2.3 seconds
            validators.RangeValidator("aggressive_ego", metrics.SimulatedVsRecordedEgoSpeedMetric,
                                      max_value=5.0, violation_duration_s=2.3,
                                      duration_mode=validators.DurationMode.CONTINUOUS),
        ]

    def get_validator_interventions(self) -> List[str]:
        """Returns a list of validators that are considered an intervention.
        """
        return [
            "displacement_error_l2",
            "distance_ref_trajectory",
            "collision_front",
            "collision_rear",
            "collision_side",
        ]

    def build_composite_metrics(self) -> List[cm.SupportsCompositeMetricCompute]:
        """Return a list of composite metrics that should be computed. Composite
        metrics are metrics that depend upon metrics and validator results."""
        interventions_val_list = ["collision_front",
                                  "collision_side",
                                  "collision_rear",
                                  "displacement_error_l2"]

        return [
            # Passed driven miles
            cm.PassedDrivenMilesCompositeMetric("passed_driven_miles_simulated",
                                                interventions_val_list,
                                                metrics.SimulatedDrivenMilesMetric,
                                                ignore_entire_scene=False),
            cm.PassedDrivenMilesCompositeMetric("passed_driven_miles_replay",
                                                interventions_val_list,
                                                metrics.ReplayDrivenMilesMetric,
                                                ignore_entire_scene=False),
            # Total driven miles
            cm.DrivenMilesCompositeMetric("total_driven_miles_simulated",
                                          metrics.SimulatedDrivenMilesMetric),
            cm.DrivenMilesCompositeMetric("total_driven_miles_replay",
                                          metrics.ReplayDrivenMilesMetric),
        ]
