import os
import time
import random

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Verify the Environment
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.benchmark import L5RasterEnv1, L5RasterEnv2

def rollout(env, n_envs, total_eps=10, total_steps=10000, monitor_eps=True):
    _ = env.reset()
    num_eps = 0
    num_steps = 0
    while True:
        obs, rewards, dones, info = env.step(n_envs * [1])
        num_steps += n_envs
        num_eps += sum(dones)

        if monitor_eps and (num_eps >= total_eps): 
            break

        if (not monitor_eps) and (num_steps >= total_steps): 
            break

    return num_steps, num_eps
  

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/home/ubuntu/level5_data/"

dm = LocalDataManager(None)
# get config
cfg = load_config_data("./config.yaml")

# rasterisation
rasterizer = build_rasterizer(cfg, dm)
raster_size = cfg["raster_params"]["raster_size"][0]
print("Raster Size: ", raster_size)

# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = EgoDataset(cfg, train_zarr, rasterizer)

if cfg["gym_params"]["load_samples"]:
    loaded_scene_dict = {}
    print("Loading 1000 sample scenes")
    for i in range(1000):
        dataset_i = train_dataset.get_scene_dataset(i)
        loaded_scene_dict[i] = {"frames": dataset_i.dataset.frames,
                                "agents": dataset_i.dataset.agents,
                                "tl_faces": dataset_i.dataset.tl_faces}
    sample_function = train_dataset.sample_function
    env = L5RasterEnv1(loaded_scene_dict, sample_function, \
                       n_channels=5, raster_size=raster_size)
else:
    env = L5RasterEnv2(train_dataset, \
                       n_channels=5, raster_size=raster_size)

# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=False)
print("Gym Env Check Passed")

# wrap it in vecEnv
n_envs = cfg["gym_params"]["num_envs"]
env = make_vec_env(lambda: env, n_envs=n_envs, \
                   vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
# env = make_vec_env(lambda: env, n_envs=n_envs)

# rollout params
total_eps = cfg["gym_params"]["total_eps"]
total_steps = cfg["gym_params"]["total_steps"]
monitor_eps = cfg["gym_params"]["monitor_eps"]

# Warm Up
if cfg["gym_params"]["warm_up"]:
    print("Warm Up")
    _, _ = rollout(env, n_envs, total_eps, total_steps, monitor_eps)

# Benchmark
print("Benchmarking")
start = time.time()
num_steps, num_eps = rollout(env, n_envs, total_eps, total_steps, monitor_eps)
time_comp = time.time() - start
print("Eps: ", num_eps, "Steps: ", num_steps)
print(f"Compute Time: {time_comp}")
