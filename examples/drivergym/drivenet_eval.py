import torch
from stable_baselines3.common.logger import Logger

from l5kit.cle.validators import ValidationCountingAggregator
from l5kit.dataset import EgoDataset
from l5kit.environment.gym_metric_set import CLEMetricSet
from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator


def calculate_ade_fde(cle_evaluator):
    scene_metric_results = cle_evaluator.metric_results()
    tot_ade = []
    tot_fde = []
    for _, metrics in scene_metric_results.items():
        disp_error = metrics['displacement_error_l2']
        tot_ade.append(torch.mean(disp_error[1:]).item())
        tot_fde.append(disp_error[-1].item())

    return sum(tot_ade) / len(tot_ade), sum(tot_fde) / len(tot_fde)


def eval_model(model: torch.nn.Module, dataset: EgoDataset, logger: Logger, output_name: str,
               num_scenes_to_unroll: int, num_simulation_steps: int = None) -> None:

    model.eval()
    torch.set_grad_enabled(False)

    # Close Loop Simulation
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                               distance_th_far=30, distance_th_close=15, num_simulation_steps=num_simulation_steps,
                               start_frame_index=0, show_info=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sim_loop = ClosedLoopSimulator(sim_cfg, dataset, device, model_ego=model, model_agents=None)

    # unroll
    scenes_to_unroll = list(range(num_scenes_to_unroll))
    sim_outs = sim_loop.unroll(scenes_to_unroll)

    # metric set
    metric_set = CLEMetricSet()
    cle_evaluator = metric_set.evaluator
    cle_evaluator.evaluate(sim_outs)
    ade, fde = calculate_ade_fde(cle_evaluator)
    validation_results = cle_evaluator.validation_results()
    agg = ValidationCountingAggregator().aggregate(validation_results)
    cle_evaluator.reset()

    d_set = output_name.split('_')[-2]
    iter_num = output_name.split('_')[-3]

    for k, v in agg.items():
        logger.record(f'{d_set}/{k}', v.item())

    logger.record(f'{d_set}/ade', round(ade, 3))
    logger.record(f'{d_set}/fde', round(fde, 3))

    # Dump log so the evaluation results are printed with the correct timestep
    logger.record("time/total timesteps", iter_num, exclude="tensorboard")
    logger.dump(iter_num)

    # TODO: Import metric compute functions from L5Kit.callback.
    # for k, v in agg.items():
    #     print(f'{d_set}/{k}', v.item())

    # print('ade: ', round(ade, 3))
    # print('fde: ', round(fde, 3))

    torch.set_grad_enabled(True)
