from typing import List

from l5kit.cle import metric_set, metrics


class L5GymCLEMetricSet(metric_set.L5MetricSet):
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
