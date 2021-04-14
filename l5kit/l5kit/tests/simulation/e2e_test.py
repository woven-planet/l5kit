import pytest
import torch

from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator
from l5kit.dataset import EgoDataset
from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from l5kit.tests.simulation.unroll_test import MockModel


@pytest.fixture(scope="function")
def simulation_evaluator() -> ClosedLoopEvaluator:
    return ClosedLoopEvaluator(EvaluationPlan(metrics=[DisplacementErrorL2Metric(),
                                                       DistanceToRefTrajectoryMetric(),
                                                       CollisionFrontMetric(),
                                                       CollisionRearMetric(),
                                                       CollisionSideMetric()],
                                              validators=[RangeValidator("displacement_error_l2_validator",
                                                                         DisplacementErrorL2Metric, max_value=30),
                                                          RangeValidator("distance_ref_trajectory_validator",
                                                                         DistanceToRefTrajectoryMetric, max_value=4),
                                                          RangeValidator("collision_front_validator",
                                                                         CollisionFrontMetric, max_value=0),
                                                          RangeValidator("collision_rear_validator",
                                                                         CollisionRearMetric, max_value=0),
                                                          RangeValidator("collision_side_validator",
                                                                         CollisionSideMetric, max_value=0),
                                                          ],
                                              composite_metrics=[],
                                              intervention_validators=["displacement_error_l2_validator",
                                                                       "distance_ref_trajectory_validator",
                                                                       "collision_front_validator",
                                                                       "collision_rear_validator",
                                                                       "collision_side_validator"]))


def test_e2e_no_models(ego_cat_dataset: EgoDataset, simulation_evaluator: ClosedLoopEvaluator) -> None:
    sim_cfg = SimulationConfig(use_ego_gt=True, use_agents_gt=True, disable_new_agents=False,
                               distance_th_far=1, distance_th_close=0)

    sim_loop = ClosedLoopSimulator(sim_cfg, ego_cat_dataset, torch.device("cpu"))
    sim_out = sim_loop.unroll(list(range(len(ego_cat_dataset.dataset.scenes))))
    simulation_evaluator.evaluate(sim_out)
    agg = ValidationCountingAggregator().aggregate(simulation_evaluator.validation_results())

    assert agg["displacement_error_l2_validator"].item() == 0
    assert agg["distance_ref_trajectory_validator"].item() == 0
    assert agg["collision_front_validator"].item() == 0
    assert agg["collision_rear_validator"].item() == 0
    assert agg["collision_side_validator"].item() == 0


@pytest.mark.parametrize("advance_x", [2.0, 0.0, 5.0])
def test_e2e_ego(ego_cat_dataset: EgoDataset, simulation_evaluator: ClosedLoopEvaluator,
                 advance_x: float) -> None:
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                               distance_th_far=1, distance_th_close=0, num_simulation_steps=20)

    ego_model = MockModel(advance_x=advance_x)

    sim_loop = ClosedLoopSimulator(sim_cfg, ego_cat_dataset, torch.device("cpu"),
                                   model_ego=ego_model)

    sim_out = sim_loop.unroll(list(range(len(ego_cat_dataset.dataset.scenes))))
    simulation_evaluator.evaluate(sim_out)
    agg = ValidationCountingAggregator().aggregate(simulation_evaluator.validation_results())

    keys = ["displacement_error_l2_validator", "distance_ref_trajectory_validator",
            "collision_front_validator", "collision_front_validator", "collision_side_validator",
            "collision_rear_validator"]

    if advance_x == 0.0:  # bumps by rear car
        key_to_trigger = "collision_rear_validator"
    elif advance_x == 2.0:  # too fast
        key_to_trigger = "distance_ref_trajectory_validator"
    elif advance_x == 5.0:  # too fast bumps into leading car
        key_to_trigger = "collision_front_validator"
    else:
        raise ValueError(f"advance_x {advance_x} not valid")

    for k in keys:
        if k == key_to_trigger:
            assert agg[k].item() == 4
        else:
            assert agg[k].item() == 0


@pytest.mark.parametrize("advance_x", [0.0, 2.0])
def test_e2e_agents(ego_cat_dataset: EgoDataset, simulation_evaluator: ClosedLoopEvaluator,
                    advance_x: float) -> None:
    sim_cfg = SimulationConfig(use_ego_gt=True, use_agents_gt=False, disable_new_agents=False,
                               distance_th_far=60, distance_th_close=30, num_simulation_steps=20,
                               start_frame_index=10)  # start from 0 as leading is rotated at 0

    agents_model = MockModel(advance_x=advance_x)

    sim_loop = ClosedLoopSimulator(sim_cfg, ego_cat_dataset, torch.device("cpu"),
                                   model_agents=agents_model)

    sim_out = sim_loop.unroll(list(range(len(ego_cat_dataset.dataset.scenes))))
    simulation_evaluator.evaluate(sim_out)
    agg = ValidationCountingAggregator().aggregate(simulation_evaluator.validation_results())

    keys = ["displacement_error_l2_validator", "distance_ref_trajectory_validator",
            "collision_front_validator", "collision_front_validator", "collision_side_validator",
            "collision_rear_validator"]

    if advance_x == 0.0:  # ego bumps in leading
        key_to_trigger = "collision_front_validator"
    elif advance_x == 2.0:  # rear bumps in ego
        key_to_trigger = "collision_rear_validator"
    else:
        raise ValueError(f"advance_x {advance_x} not valid")

    for k in keys:
        if k == key_to_trigger:
            assert agg[k].item() == 4
        else:
            assert agg[k].item() == 0
