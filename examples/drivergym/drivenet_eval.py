from typing import Optional

import torch
from l5kit.cle.composite_metrics import CompositeMetricAggregator
from l5kit.cle.scene_type_agg import compute_cle_scene_type_aggregations
from l5kit.cle.validators import ValidationCountingAggregator
from l5kit.dataset import EgoDataset
from l5kit.environment.callbacks import L5KitEvalCallback
from l5kit.environment.gym_metric_set import CLEMetricSet
from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from stable_baselines3.common.logger import Logger


def eval_model(model: torch.nn.Module, dataset: EgoDataset, logger: Logger, d_set: str, iter_num: int,
               num_scenes_to_unroll: int, num_simulation_steps: int = None,
               enable_scene_type_aggregation: Optional[bool] = False,
               scene_id_to_type_path: Optional[str] = None) -> None:
    """Evaluator function for the drivenet model. Evaluate the model using the CLEMetricSet
    of L5Kit. Logging is performed in the Tensorboard logger.

    :param model: the trained model to evaluate
    :param dataset: the dataset on which the models is evaluated
    :param logger: tensorboard logger to log the evaluation results
    :param d_set: the type of dataset being evaluated ("train" or "eval")
    :param iter_num: iteration number of training (to log in tensorboard)
    :param num_scenes_to_unroll: Number of scenes to evaluate in the dataset
    :param num_simulation_steps: Number of steps to unroll the model for.
    :param enable_scene_type_aggregation: enable evaluation according to scene type
    :param scene_id_to_type_path: path to the csv file mapping scene id to scene type
    """

    model.eval()
    torch.set_grad_enabled(False)

    # Close Loop Simulation
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                               distance_th_far=30, distance_th_close=15, num_simulation_steps=num_simulation_steps,
                               start_frame_index=0, show_info=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sim_loop = ClosedLoopSimulator(sim_cfg, dataset, device, model_ego=model, model_agents=None)

    # metric set
    metric_set = CLEMetricSet()

    # unroll
    batch_unroll = 100
    for start_idx in range(0, num_scenes_to_unroll, batch_unroll):
        end_idx = min(num_scenes_to_unroll, start_idx + batch_unroll)
        scenes_to_unroll = list(range(start_idx, end_idx))
        sim_outs = sim_loop.unroll(scenes_to_unroll)
        metric_set.evaluator.evaluate(sim_outs)

    # Aggregate metrics (ADE, FDE)
    ade, fde = L5KitEvalCallback.compute_ade_fde(metric_set)
    logger.record(f'{d_set}/ade', round(ade, 3))
    logger.record(f'{d_set}/fde', round(fde, 3))

    # Aggregate validators
    validation_results = metric_set.evaluator.validation_results()
    agg = ValidationCountingAggregator().aggregate(validation_results)
    for k, v in agg.items():
        logger.record(f'{d_set}/{k}', v.item())
    # Add total collisions as well
    tot_collision = agg['collision_front'].item() + agg['collision_side'].item() + agg['collision_rear'].item()
    logger.record(f'{d_set}/total_collision', tot_collision)

    # Aggregate composite metrics
    composite_metric_results = metric_set.evaluator.composite_metric_results()
    comp_agg = CompositeMetricAggregator().aggregate(composite_metric_results)
    for k, v in comp_agg.items():
        logger.record(f'{d_set}/{k}', v.item())

    # If we should compute the scene-type aggregation metrics
    if enable_scene_type_aggregation:
        assert scene_id_to_type_path is not None
        scene_ids_to_scene_types = L5KitEvalCallback.get_scene_types(scene_id_to_type_path)
        scene_type_results = \
            compute_cle_scene_type_aggregations(metric_set,
                                                scene_ids_to_scene_types,
                                                list_validator_table_to_publish=[])
        for k, v in scene_type_results.items():
            logger.record(f'{k}', v)

    # Dump log so the evaluation results are printed with the correct timestep
    logger.record("time/total timesteps", iter_num, exclude="tensorboard")
    logger.dump(iter_num)

    metric_set.evaluator.reset()
    torch.set_grad_enabled(True)
