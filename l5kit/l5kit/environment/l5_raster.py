import os
import time
import random
import pickle
from typing import List, Dict, Optional, Tuple
import argparse

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv

# Verify the Environment
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.benchmark import L5RasterBaseEnv, L5RasterCacheEnv, L5DatasetCacheEnv

def rollout(env: VecEnv, n_envs: int, total_eps: int = 10, total_steps: int = 10000,
            monitor_eps: bool = True)-> Tuple[int, int]:
    """Collect experiences using the current policy. Dummy actions used.
       The term rollout here refers to the model-free notion and should not
       be used with the concept of rollout used in model-based RL or planning.
    
    :param env: The training environment
    :param n_envs: the number of parallel gym envts
    :param total_steps: Number of experiences (in terms of steps) to collect per environment
    :param total_eps: Number of experiences (in terms of episodes) to collect per environment
    :param monitor_eps: flag to terminate rollout based on total_steps or total_eps.
    :return: the tuple of [num steps rolled out, num episodes rolled out]
    """
    _ = env.reset()
    num_eps = 0
    num_steps = 0

    dummy_act = n_envs * [1]
    if n_envs == 0:
        dummy_act = 1

    while True:
        obs, rewards, dones, info = env.step(dummy_act)
        num_steps += n_envs
        if n_envs == 0:
            dones = [dones]

        num_eps += sum(dones)
        if sum(dones):
            if num_eps % 10 == 0:
                print(num_eps)
            if n_envs == 0:
                obs = env.reset()

        if monitor_eps and (num_eps >= total_eps): 
            break

        if (not monitor_eps) and (num_steps >= total_steps): 
            break

    return num_steps, num_eps


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_envs', default=None, type=int,
                        help='Number of parallel envts')
    parser.add_argument('--env_class', default=None, type=str,
                        choices=('Dummy', 'SubProc', 'Main'),
                        help='env_class')
    parser.add_argument('--use_ego_model', action='store_true',
                        help='Use ego model for actions')
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

    # init gym environment
    # Pre-compute rasters and save to cache
    if cfg["gym_params"]["pre_compute_rasters"]:
        try:
            # open already saved rasters
            with open('raster_dict.pkl', 'rb') as f:
                loaded_scene_dict = pickle.load(f)
        except:
            loaded_scene_dict = {}
            sample_function = train_dataset.sample_function
            print("Pre-computing 100 sample scene rasters")
            print("This may take some time...............")
            for i in range(100):
                dataset_i = train_dataset.get_scene_dataset(i)
                loaded_scene_dict[i] = {"frames": dataset_i.dataset.frames,
                                        "agents": dataset_i.dataset.agents,
                                        "tl_faces": dataset_i.dataset.tl_faces}
                # generate raster for all the frames
                raster_data = {}
                for frame_id in range(len(loaded_scene_dict[i]["frames"])):
                    data = sample_function(frame_id, loaded_scene_dict[i]["frames"], \
                                        loaded_scene_dict[i]["agents"], \
                                        loaded_scene_dict[i]["tl_faces"], None)
                    raster_data[frame_id] = data["image"].transpose(2, 0, 1)
                loaded_scene_dict[i]["rasters"] = raster_data
            # Save generated rasters
            with open('raster_dict.pkl', 'wb') as f:
                pickle.dump(loaded_scene_dict, f, pickle.HIGHEST_PROTOCOL)

        # initial Gym envt with cached rasters
        env = L5RasterCacheEnv(loaded_scene_dict, rasterizer)

    # Pre-load dataset into cache
    elif cfg["gym_params"]["pre_load_dataset"]:
        loaded_scene_dict = {}
        print("Loading 100 sample scenes")
        for i in range(100):
            dataset_i = train_dataset.get_scene_dataset(i)
            loaded_scene_dict[i] = {"frames": dataset_i.dataset.frames,
                                    "agents": dataset_i.dataset.agents,
                                    "tl_faces": dataset_i.dataset.tl_faces}

        # initial Gym envt with cached dataset
        env = L5DatasetCacheEnv(loaded_scene_dict, rasterizer)
        env.sample_function = train_dataset.sample_function

    else:
        num_simulation_steps = cfg["gym_params"]["num_simulation_steps"]
        # initial Base Gym envt
        env = L5RasterBaseEnv(train_dataset, rasterizer, num_simulation_steps)

    # If the environment don't follow the interface, an error will be thrown
    if cfg["gym_params"]["check_env"]:
        check_env(env, warn=False)
        print("Custom Gym Environment Check Passed")
        exit()

    # wrap it in vecEnv
    n_envs = args.n_envs if args.n_envs is not None else cfg["gym_params"]["num_envs"]
    env_class = args.env_class if args.env_class is not None else cfg["gym_params"]["env_class"]
    print("Number of Parallel Enviroments: ", n_envs, " ", env_class)
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

    # Warm Up
    if cfg["gym_params"]["warm_up"]:
        print("Warm Up")
        _, _ = rollout(env, n_envs, 20, total_steps, monitor_eps)

    # Benchmark
    print("Benchmarking")
    start = time.time()
    num_steps, num_eps = rollout(env, n_envs, total_eps, total_steps, monitor_eps)
    time_comp = time.time() - start
    print("Eps: ", num_eps, "Steps: ", num_steps)
    print(f"Compute Time: {time_comp}")
