import os
import time
import random
import pickle
from typing import List, Dict, Optional, Tuple
import argparse

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
from l5kit.benchmark import L5Env
from l5kit.benchmark.cle_utils import get_cle, calculate_cle_metrics, SimulationOutputGym, aggregate_cle_metrics


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

    if n_envs == 0:
        return action

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
    # obs['image'] = torch.Tensor(obs['image'])
    # obs['image'] = [n_envs, 15, 224, 224]
    obs['image'] = torch.from_numpy(obs['image']).float()
    if n_envs == 0:
        obs['image'] = obs['image'].unsqueeze(0)

    # act['positions'] = [n_envs, 12, 2]
    # act['yaws'] = [n_envs, 12, 1]
    # act['velocities'] = [n_envs, 12, 2]
    act = ego_model(move_to_device(obs, device))
    act = move_to_numpy(act)

    if n_envs == 0:
        obs['image'] = obs['image'].squeeze(0)
    obs['image'] = obs['image'].cpu().numpy()

    if n_envs == 0:
        return act

    # action_list: 
    # 0: ['positions'] = [12, 2]; ['yaws'] = [12, 1]; ['velocities'] = [12, 2]  
    # 1: ['positions'] = [12, 2]; ['yaws'] = [12, 1]; ['velocities'] = [12, 2]  
    # .....
    # n_envs: ['positions'] = [12, 2]; ['yaws'] = [12, 1]; ['velocities'] = [12, 2]  
    action_list = []
    for n in range(n_envs):
        action_list.append({'positions': act['positions'][n:n+1], 
                            'yaws': act['yaws'][n:n+1],
                            'velocities': act['velocities'][n:n+1]})
    return action_list


def rollout(env: VecEnv, n_envs: int, total_eps: int = 10, total_steps: int = 10000,
            monitor_eps: bool = True, future_num_frames: int = 12,
            use_ego_model: bool = False,
            ego_model: Optional[torch.nn.Module] = None,
            device: Optional[torch.device] = None)-> Tuple[int, int, List[SimulationOutputGym]]:
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
    print("Warning: Dummy Observation being used for the Ego Model")
    dummy_obs = {"image": np.random.normal(loc=0.0, scale=1.0, size=(15, 224, 224))}    
    if n_envs:
        dummy_obs = {"image": np.random.normal(loc=0.0, scale=1.0, size=(n_envs, 15, 224, 224))}
    obs = env.reset()
    num_eps = 0
    num_steps = 0

    # device = device if not None else torch.device("cpu") 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not use_ego_model:
        ego_model = None
    if ego_model is not None:
        # act = get_model_action(obs, ego_model, device, n_envs)
        act = get_model_action(dummy_obs, ego_model, device, n_envs)
    else:
        act = get_dummy_action(n_envs, future_num_frames)

    while True:
        obs, rewards, dones, info = env.step(act)
        num_steps += n_envs
        # print("Steps:", num_steps)

        if n_envs == 0:
            dones = [dones]

        num_eps += sum(dones)
        if sum(dones):
            if num_eps % 20 == 0:
                print(num_eps)
            if n_envs == 0:
                obs = env.reset()

        if ego_model is not None:
            # act = get_model_action(obs, ego_model, device, n_envs)
            act = get_model_action(dummy_obs, ego_model, device, n_envs)

        if monitor_eps and (num_eps >= total_eps): 
            break

        if (not monitor_eps) and (num_steps >= total_steps): 
            break

    if n_envs == 0:
        return num_steps, num_eps, env.cle_evaluator
    return num_steps, num_eps, env.get_attr("cle_evaluator")[0]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_envs', default=None, type=int,
                        help='Number of parallel envts')
    parser.add_argument('--env_class', default=None, type=str,
                        choices=('Dummy', 'SubProc', 'Main'),
                        help='env_class')
    parser.add_argument('--use_ego_model', action='store_true',
                        help='Use ego model for actions')
    parser.add_argument('--out_size', default=None, type=int,
                        help='Output Raster Size for IPC')
    args = parser.parse_args()

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    ego_model_path = "/home/ubuntu/models/planning_model_20210421_5steps.pt"
    ego_model = torch.load(ego_model_path).to(device)
    ego_model = ego_model.eval()
    torch.set_grad_enabled(False)
    use_ego_model = args.use_ego_model if args.use_ego_model is not None else cfg["gym_params"]["use_ego_model"]
    print("Use Ego Model: ", use_ego_model)

    # get closed_loop_evaluator
    cle_evaluator = get_cle()

    # init environment
    future_num_frames = ego_model.model.fc.out_features // 3 # X, Y, Yaw
    num_simulation_steps = cfg["gym_params"]["num_simulation_steps"]
    det_roll = cfg["gym_params"]["deterministic_rollout"]
    raster_out_size = args.out_size if args.out_size is not None else cfg["gym_params"]["raster_out_size"]
    env = L5Env(train_dataset, rasterizer, future_num_frames, None,
                cle_evaluator, det_roll, raster_out_size)

    # If the environment don't follow the interface, an error will be thrown
    if cfg["gym_params"]["check_env"]:
        check_env(env, warn=False)
        print("Custom Gym Environment Check Passed")
        exit()

    # wrap it in vecEnv
    n_envs = args.n_envs if args.n_envs is not None else cfg["gym_params"]["num_envs"]
    env_class = args.env_class if args.env_class is not None else cfg["gym_params"]["env_class"]
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
    # No VecEnv
    else:
        assert n_envs == 0, "VecEnvironment Not Implemented"
        # raise NotImplementedError

    # rollout params
    total_eps = cfg["gym_params"]["total_eps"]
    total_steps = cfg["gym_params"]["total_steps"]
    monitor_eps = cfg["gym_params"]["monitor_eps"]

    # WarmUp. No Warm Up for deterministic rollout
    if cfg["gym_params"]["warm_up"] and (not det_roll):
        print("Warm Up")
        _, _, _ = rollout(env, n_envs, 40, total_steps, monitor_eps, future_num_frames,
                       use_ego_model, ego_model, device)

    # Benchmark
    print("Benchmark")
    start = time.time()
    num_steps, num_eps, cle_evaluator = rollout(env, n_envs, total_eps, total_steps, monitor_eps, future_num_frames,
                                                use_ego_model, ego_model, device)
    time_comp = time.time() - start
    print("Eps: ", num_eps, "Steps: ", num_steps)
    print(f"Compute Time: {time_comp}")

    # Calculate CLE metrics if deterministic rollout
    if det_roll:
        # calculate_cle_metrics(sim_outs_log)
        aggregate_cle_metrics(cle_evaluator)