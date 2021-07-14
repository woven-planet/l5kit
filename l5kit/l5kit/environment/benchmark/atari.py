""" Benchmark Atari Games Rasterizer """
import time
from typing import Tuple

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecFrameStack

from l5kit.configs import load_config_data


def rollout(env: VecEnv, n_envs: int, total_eps: int = 10, total_steps: int = 10000,
            monitor_eps: bool = True) -> Tuple[int, int]:
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
    dummy_action = n_envs * [1]
    while True:
        obs, rewards, dones, info = env.step(dummy_action)
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
n_envs = cfg["atari_params"]["num_envs"]

# Note: Download Atari ROMS. Instructions: https://github.com/openai/atari-py#roms
# Atari ID examples: PongNoFrameskip-v4 (default), SpaceInvadersNoFrameskip-v4.
env = make_atari_env(cfg["atari_params"]["atari_id"], n_envs=n_envs, seed=0,
                     vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": 'fork'})

# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

# rollout params
total_eps = cfg["atari_params"]["total_eps"]
total_steps = cfg["atari_params"]["total_steps"]
monitor_eps = cfg["atari_params"]["monitor_eps"]

# Warm Up
if cfg["atari_params"]["warm_up"]:
    print("Warm Up")
    _, _ = rollout(env, n_envs, total_eps, total_steps, monitor_eps)

# Benchmark
print("Benchmarking")
start = time.time()
num_steps, num_eps = rollout(env, n_envs, total_eps, total_steps, monitor_eps)
time_comp = time.time() - start
print("Eps: ", num_eps, "Steps: ", num_steps)
print(f"Compute Time: {time_comp}")
