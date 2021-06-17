import os
import time

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


class L5Raster(gym.Env):
    """
    Custom Environment of L5 Kit that follows gym interface.
    This is a simple env where a random frame is returned as observation.
    """

    def __init__(self, input_dataset, n_channels, raster_size):
        super(L5Raster, self).__init__()

        self.input_dataset = input_dataset
        self.scene_id = 0
        self.raster_size = raster_size

        # Define action and observation space
        # They must be gym.spaces objects
        # DUMMY Action Space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(n_channels, raster_size, raster_size), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.scene_id = 0
        obs = self.input_dataset[self.scene_id]["image"]
        return obs

    def step(self, action):
        obs = self.input_dataset[self.scene_id]["image"]
        self.scene_id += 1

        # done always set to False
        done = False

        # reward always set to 1
        reward = 1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        # return np.array([self.agent_pos]).astype(np.float32), reward, done, info
        return obs, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass


# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/home/ubuntu/level5_data/"

dm = LocalDataManager(None)
# get config
cfg = load_config_data("./config.yaml")

# rasterisation
rasterizer = build_rasterizer(cfg, dm)

# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = EgoDataset(cfg, train_zarr, rasterizer)

print("Raster Size: ", cfg["raster_params"]["raster_size"][0])
env = L5Raster(train_dataset, n_channels=5, raster_size=cfg["raster_params"]["raster_size"][0])
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=False)
print("Gym Env Check Passed")

# wrap it
n_envs = cfg["gym_params"]["num_envs"]
env = make_vec_env(lambda: env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
# env = make_vec_env(lambda: env, n_envs=n_envs)

num_samples = cfg["gym_params"]["num_samples"]
obs = env.reset()
# Warm Up
print("Warm Up")
for _ in range(num_samples // n_envs):
    obs, rewards, dones, info = env.step(n_envs * [1])

# Benchmark
print("Benchmarking")
start = time.time()
for _ in range(num_samples // n_envs):
    obs, rewards, dones, info = env.step(n_envs * [1])
time_comp = time.time() - start
print(f"Compute Time: {time_comp}")
