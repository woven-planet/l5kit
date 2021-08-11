from typing import List

from l5kit.cle import metric_set, metrics


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
