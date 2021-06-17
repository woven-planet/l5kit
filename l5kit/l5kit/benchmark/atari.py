import time

import numpy as np
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from l5kit.configs import load_config_data


# get config
cfg = load_config_data("./config.yaml")

# There already exists an environment generator that will make and wrap atari environments correctly.
n_envs = cfg["gym_params"]["num_envs"]

# Note: Download Atari ROMS. Instructions: https://github.com/openai/atari-py#roms
# Atari ID examples: PongNoFrameskip-v4 (default), SpaceInvadersNoFrameskip-v4.
env = make_atari_env(cfg["gym_params"]["atari_id"], n_envs=n_envs,
                     seed=0, wrapper_kwargs={"frame_skip": 4},
                     vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": 'fork'})

# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

obs = env.reset()
# Random Action
rand_act = np.array(n_envs * [env.action_space.sample()])
num_samples = cfg["gym_params"]["num_samples"]

# Warm Up
print("Warm Up")
for _ in range(num_samples // n_envs):
    obs, rewards, dones, info = env.step(rand_act)

# Benchmark
print("Benchmarking")
start = time.time()
for _ in range(num_samples // n_envs):
    obs, rewards, dones, info = env.step(rand_act)
time_comp = time.time() - start
print(f"Compute Time: {time_comp}")
