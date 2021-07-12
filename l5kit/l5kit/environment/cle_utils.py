from typing import List, DefaultDict

from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                            DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric, SimulatedDrivenMilesMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator
from l5kit.simulation.unroll import SimulationOutput, UnrollInputOutput
from l5kit.simulation.dataset import SimulationDataset
from prettytable import PrettyTable

def get_cle() -> ClosedLoopEvaluator:
    # metrics = [DisplacementErrorL2Metric(),
    #         DistanceToRefTrajectoryMetric(),
    #         CollisionFrontMetric(),
    #         CollisionRearMetric(),
    #         CollisionSideMetric()]

    # validators = [RangeValidator("displacement_error_l2_validator", DisplacementErrorL2Metric, max_value=5),
    #             RangeValidator("distance_ref_trajectory_validator", DistanceToRefTrajectoryMetric, max_value=1),
    #             RangeValidator("collision_front_validator", CollisionFrontMetric, max_value=0),
    #             RangeValidator("collision_rear_validator", CollisionRearMetric, max_value=0),
    #             RangeValidator("collision_side_validator", CollisionSideMetric, max_value=0),
    #             ]

    # intervention_validators = ["displacement_error_l2_validator",
    #                         "distance_ref_trajectory_validator",
    #                         "collision_front_validator",
    #                         "collision_rear_validator",
    #                         "collision_side_validator"]

    metrics = [DisplacementErrorL2Metric(),
            DistanceToRefTrajectoryMetric(scene_fraction=1.0),
            SimulatedDrivenMilesMetric()
            ]

    validators = [RangeValidator("displacement_error_l2_validator", DisplacementErrorL2Metric, max_value=5),
                RangeValidator("distance_ref_trajectory_validator", DistanceToRefTrajectoryMetric, max_value=1),
                ]

    intervention_validators = ["displacement_error_l2_validator",
                            "distance_ref_trajectory_validator"]

    # cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
    #                                     validators=validators,
    #                                     composite_metrics=[],
    #                                     intervention_validators=intervention_validators))
    cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
                                        validators=[],
                                        composite_metrics=[],
                                        intervention_validators=[]))
    return cle_evaluator

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

def aggregate_cle_metrics(cle_evaluator: ClosedLoopEvaluator) -> None:
    validation_results_log = cle_evaluator.validation_results()
    agg_log = ValidationCountingAggregator().aggregate(validation_results_log)
    cle_evaluator.reset()

    fields = ["metric", "log_replayed agents"]
    table = PrettyTable(field_names=fields)
    for metric_name in agg_log:
        table.add_row([metric_name, agg_log[metric_name].item()])
    print(table)

class SimulationOutputGym(SimulationOutput):
    def __init__(self, scene_id: int, sim_dataset: SimulationDataset,
                 ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]],
                 agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]]):
        """This object holds information about the result of the simulation loop
        for a given scene dataset in Gym

        :param scene_id: the scene indices
        :param sim_dataset: the simulation dataset
        :param ego_ins_outs: all inputs and outputs for ego (each frame of each scene has only one)
        :param agents_ins_outs: all inputs and outputs for agents (multiple per frame in a scene)
        """
        SimulationOutput.__init__(self, scene_id, sim_dataset, ego_ins_outs, agents_ins_outs)

        # Required for Bokeh Visualizer
        self.tls_frames = self.simulated_dataset.dataset.tl_faces
        self.agents_th = self.simulated_dataset.cfg["raster_params"]["filter_agents_threshold"]

        # Remove Dataset attributes
        self.recorded_dataset = None
        self.simulated_dataset = None
