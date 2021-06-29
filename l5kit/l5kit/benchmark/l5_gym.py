import os
import time
import random
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

# Verify the Environment
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer, Rasterizer
from l5kit.dataset.utils import move_to_device, move_to_numpy
from l5kit.simulation.unroll import SimulationOutput
from l5kit.benchmark import L5Env
from l5kit.benchmark.cle_utils import calculate_cle_metrics

def get_dummy_action(n_envs: int, future_num_frames: int) -> List[Dict[str, np.ndarray]]:
    """This function outputs dummy action for the gym envt.

    :param n_envs: the number of parallel gym envts
    :param future_num_frames: the number of frames to predict
    :return: the list of dict mapping prediction attributes to dummy values
    """
    action = {}
    action["positions"] = np.zeros((1, future_num_frames, 2))
    action["yaws"] = np.zeros((1, future_num_frames, 1))
    action["velocities"] = np.zeros((1, future_num_frames, 2))

    action_list = []
    for _ in range(n_envs):
        action_list.append(action)
    return action_list


def get_model_action(obs: Dict[str, np.ndarray],
                     ego_model: torch.nn.Module,
                     device: torch.device,
                     n_envs: int) -> List[Dict[str, np.ndarray]]:
    """This function outputs action for the gym envt provided by trained ego_model.

    :param obs: the observation of gym envt.
    :param ego_model: the model to be used for ego
    :param device: a torch device. Inference will be performed here
    :param n_envs: the number of parallel gym envts
    :return: the list of dict mapping prediction attributes to ego_model outputs
    """
    obs['image'] = torch.Tensor(obs['image'])
    act = ego_model(move_to_device(obs, device))

    action_list = []
    for n in range(n_envs):
        action_list.append({'positions': act['positions'][n:n+1].cpu().numpy(), 
                            'yaws': act['yaws'][n:n+1].cpu().numpy(),
                            'velocities': act['velocities'][n:n+1].cpu().numpy()})
    return action_list


def rollout(env: VecEnv, n_envs: int, total_eps: int = 10, total_steps: int = 10000,
            monitor_eps: bool = True, future_num_frames: int = 12,
            use_ego_model: bool = False,
            ego_model: Optional[torch.nn.Module] = None,
            device: Optional[torch.device] = None)-> Tuple[int, int, List[SimulationOutput]]:
    """Collect experiences using the current policy and fill a ``RolloutBuffer`` (TBD).
       The term rollout here refers to the model-free notion and should not
       be used with the concept of rollout used in model-based RL or planning.
    
    :param env: The training environment
    :param n_envs: the number of parallel gym envts
    :param total_steps: Number of experiences (in terms of steps) to collect per environment
    :param total_eps: Number of experiences (in terms of episodes) to collect per environment
    :param monitor_eps: flag to terminate rollout based on total_steps or total_eps.
    :param future_num_frames: the number of frames to predict
    :param use_ego_model: flag to use the ego model
    :param ego_model: the model to be used for ego
    :param device: a torch device. Inference will be performed here
    :return: the tuple of [num steps rolled out, num episodes rolled out]
    """
    obs = env.reset()
    num_eps = 0
    num_steps = 0
    sim_logs = []

    device = device if not None else torch.device("cpu") 
    if not use_ego_model:
        ego_model = None
    if ego_model is not None:
        act = get_model_action(obs, ego_model, device, n_envs)
    else:
        act = get_dummy_action(n_envs, future_num_frames)

    while True:
        obs, rewards, dones, info = env.step(act)
        num_steps += n_envs
        # print("Steps:", num_steps)
        num_eps += sum(dones)

        if ego_model is not None:
            act = get_model_action(obs, ego_model, device, n_envs)

        if dones[0] and env.get_attr("det_roll")[0]:
            print("Done Done:", dones)
            sim_logs.append(info[0]["info"][0])

        if monitor_eps and (num_eps >= total_eps): 
            break

        if (not monitor_eps) and (num_steps >= total_steps): 
            break

    return num_steps, num_eps, sim_logs


if __name__=="__main__":
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = "/home/ubuntu/level5_data/"

    dm = LocalDataManager(None)
    # get config
    cfg = load_config_data("./config_nb.yaml")

    # rasterisation
    rasterizer = build_rasterizer(cfg, dm)
    raster_size = cfg["raster_params"]["raster_size"][0]
    print("Raster Size: ", raster_size)

    # init dataset
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    train_dataset = EgoDataset(cfg, train_zarr, rasterizer)

    # load pretrained model
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    ego_model_path = "/home/ubuntu/models/planning_model_20210421_5steps.pt"
    ego_model = torch.load(ego_model_path).to(device)
    ego_model = ego_model.eval()
    torch.set_grad_enabled(False)
    use_ego_model = cfg["gym_params"]["use_ego_model"]

    # init environment
    future_num_frames = ego_model.model.fc.out_features // 3 # X, Y, Yaw
    num_simulation_steps = cfg["gym_params"]["num_simulation_steps"]
    det_roll = cfg["gym_params"]["deterministic_rollout"]
    env = L5Env(train_dataset, rasterizer, future_num_frames, num_simulation_steps, det_roll)

    # If the environment don't follow the interface, an error will be thrown
    if cfg["gym_params"]["check_env"]:
        check_env(env, warn=False)
        print("Custom Gym Environment Check Passed")
        exit()

    # wrap it in vecEnv
    n_envs = cfg["gym_params"]["num_envs"]
    env_class = cfg["gym_params"]["env_class"]
    print("Number of Parallel Enviroments: ", n_envs, " ", env_class)

    if det_roll:
        assert n_envs == 1, "Number of Envts should be 1 for deterministic rollout"
        assert env_class == "Dummy", "Envt Class should be Dummy for deterministic rollout"

    # SubProcVecEnv
    if env_class == 'SubProc':
        env = make_vec_env(lambda: env, n_envs=n_envs, \
                           vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
    # DummyVecEnv
    elif env_class == 'Dummy':
        env = make_vec_env(lambda: env, n_envs=n_envs)
    else:
        raise NotImplementedError

    # rollout params
    total_eps = cfg["gym_params"]["total_eps"]
    total_steps = cfg["gym_params"]["total_steps"]
    monitor_eps = cfg["gym_params"]["monitor_eps"]

    # WarmUp. No Warm Up for deterministic rollout
    if cfg["gym_params"]["warm_up"] and (not det_roll):
        print("Warm Up")
        _, _, _ = rollout(env, n_envs, total_eps, total_steps, monitor_eps, future_num_frames,
                       use_ego_model, ego_model, device)

    # Benchmark
    print("Benchmark")
    start = time.time()
    num_steps, num_eps, sim_outs_log = rollout(env, n_envs, total_eps, total_steps, monitor_eps, future_num_frames,
                                               use_ego_model, ego_model, device)
    time_comp = time.time() - start
    print("Eps: ", num_eps, "Steps: ", num_steps)
    print(f"Compute Time: {time_comp}")

    # Calculate CLE metrics if deterministic rollout
    if det_roll:
        calculate_cle_metrics(sim_outs_log)
