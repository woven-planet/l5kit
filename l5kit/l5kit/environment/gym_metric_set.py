from typing import List

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


class L2DisplacementYawCollisionMetricSet(metric_set.L5MetricSet):
    """This class is responsible for computing a set of metric parametrization
    for the L5Kit Gym-compatible environment. The metrics in this class comprise
    L2 displacement error for the (x, y) coordinates, closest-angle error for the yaw
    and collision.

    :param metric_prefix: this is a prefix that will identify the metric set being used.
    """

    def __init__(self, metric_prefix: str = "L2DisplacementYawCollision") -> None:
        """Constructor method
        """
        super().__init__(metric_prefix)

    def build_metrics(self) -> List[metrics.SupportsMetricCompute]:
        """Returns a list of metrics that will be computed.
        """
        return [
            metrics.DisplacementErrorL2Metric(),
            metrics.YawErrorCAMetric(),
            metrics.CollisionFrontMetric(),
            metrics.CollisionRearMetric(),
            metrics.CollisionSideMetric(),
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
            metrics.CollisionSideMetric()
        ]

    def build_validators(self) -> List[validators.SupportsMetricValidate]:
        """Returns a list of validators that will operate on the computed metrics.
        """
        return [
            validators.RangeValidator("displacement_error_l2", metrics.DisplacementErrorL2Metric, max_value=30),
            validators.RangeValidator("distance_ref_trajectory", metrics.DistanceToRefTrajectoryMetric, max_value=4),
            validators.RangeValidator("collision_front", metrics.CollisionFrontMetric, max_value=0),
            validators.RangeValidator("collision_rear", metrics.CollisionRearMetric, max_value=0),
            validators.RangeValidator("collision_side", metrics.CollisionSideMetric, max_value=0)
        ]

    def get_validator_interventions(self) -> List[str]:
        """Returns a list of validators that are considered an intervention.
        """
        return [
            "displacement_error_l2",
            "distance_ref_trajectory",
            "collision_front",
            "collision_rear",
            "collision_side"
        ]
