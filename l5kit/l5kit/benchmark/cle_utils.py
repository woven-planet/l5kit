from typing import List

from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                            DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator
from l5kit.simulation.unroll import SimulationOutput
from prettytable import PrettyTable

metrics = [DisplacementErrorL2Metric(),
        DistanceToRefTrajectoryMetric(),
        CollisionFrontMetric(),
        CollisionRearMetric(),
        CollisionSideMetric()]

validators = [RangeValidator("displacement_error_l2_validator", DisplacementErrorL2Metric, max_value=5),
            RangeValidator("distance_ref_trajectory_validator", DistanceToRefTrajectoryMetric, max_value=1),
            RangeValidator("collision_front_validator", CollisionFrontMetric, max_value=0),
            RangeValidator("collision_rear_validator", CollisionRearMetric, max_value=0),
            RangeValidator("collision_side_validator", CollisionSideMetric, max_value=0),
            ]

intervention_validators = ["displacement_error_l2_validator",
                        "distance_ref_trajectory_validator",
                        "collision_front_validator",
                        "collision_rear_validator",
                        "collision_side_validator"]

cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
                                    validators=validators,
                                    composite_metrics=[],
                                    intervention_validators=intervention_validators))

def calculate_cle_metrics(sim_outs_log: List[SimulationOutput]) -> None:
    cle_evaluator.evaluate(sim_outs_log)
    validation_results_log = cle_evaluator.validation_results()
    agg_log = ValidationCountingAggregator().aggregate(validation_results_log)
    cle_evaluator.reset()

    fields = ["metric", "log_replayed agents"]
    table = PrettyTable(field_names=fields)
    for metric_name in agg_log:
        table.add_row([metric_name, agg_log[metric_name].item()])
    print(table)
