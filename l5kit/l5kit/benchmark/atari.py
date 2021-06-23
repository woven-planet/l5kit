import time

import numpy as np
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from l5kit.configs import load_config_data

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
