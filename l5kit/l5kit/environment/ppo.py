import argparse
import os
import time
from typing import Any, Optional

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.environment import feature_extractor
from l5kit.environment.callbacks import LoggingCallback, VizCallback
from l5kit.environment.envs.l5_env import SimulationConfigGym


def get_callback_list(output_prefix: str, n_envs: int, save_freq: Optional[int] = 50000,
                      args: Optional[Any] = None, ckpt_prefix: Optional[str] = None) -> CallbackList:

    callback_list = []
    # Define callbacks
    assert output_prefix is not None, "Provide output prefix to save model states"

    print(f"Saving Model every {save_freq} steps")
    # Save SimulationOutputGym periodically
    viz_callback = VizCallback(save_freq=(save_freq // n_envs), save_path='./logs/',
                               name_prefix=output_prefix, verbose=2)
    callback_list.append(viz_callback)

    # Save Model Periodically
    ckpt_prefix = ckpt_prefix if ckpt_prefix is not None else output_prefix
    checkpoint_callback = CheckpointCallback(save_freq=(save_freq // n_envs), save_path='./logs/',
                                             name_prefix=ckpt_prefix, verbose=2)
    callback_list.append(checkpoint_callback)

    # Save Model Config
    log_callback = LoggingCallback(args)
    callback_list.append(log_callback)

    # Print Error at end of OpenLoop training
    # if not clt:
    #     traj_callback = TrajectoryCallback(save_path='./logs/', verbose=2)
    #     callback_list.append(traj_callback)

    callback = CallbackList(callback_list)
    return callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_prefix', required=True, type=str,
                        help='Output prefix for tracking model outputs during training')
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/level5_data/',
                        help='Environment configuration file')
    parser.add_argument('--env_config_path', type=str, default='/home/ubuntu/src/l5kit/examples/RL/config.yaml',
                        help='Environment configuration file')
    parser.add_argument('--disable_cle', action='store_true',
                        help='Flag to disable close loop environment')
    parser.add_argument('--save_freq', default=50000, type=int,
                        help='Frequency to save model states')
    parser.add_argument('--n_envs', default=4, type=int,
                        help='Number of parallel envts')
    parser.add_argument('--eps_length', default=16, type=int,
                        help='Number of time steps')
    parser.add_argument('--n_steps', default=3000, type=int,
                        help='Number of time steps')
    parser.add_argument('--num_rollout_steps', default=64, type=int,
                        help='Number of rollout steps per update')
    parser.add_argument('--n_epochs', default=10, type=int,
                        help='Number of model epochs per update')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Discount factor')
    parser.add_argument('--tb_log', default=None, type=str,
                        help='Tensorboard Log folder')
    parser.add_argument('--eval', action='store_true',
                        help='Eval model at end of training')
    parser.add_argument('--ckpt_prefix', default=None, type=str,
                        help='Ckpt prefix for saving model')
    parser.add_argument('--device', default="auto", type=str,
                        help='Device for running model')
    parser.add_argument('--rew_clip', default=15, type=float,
                        help='Reward Clipping Threshold')
    args = parser.parse_args()

    # By setting the L5KIT_DATA_FOLDER variable, we can point the script to the folder where the data lies.
    # The path to 'l5_data_path' needs to be provided
    os.environ["L5KIT_DATA_FOLDER"] = args.data_path

    # make gym env
    if args.n_envs == 1:
        print("Using 1 envt")
        env = gym.make("L5-CLE-v0", sim_cfg=SimulationConfigGym(args.eps_length))

    # custom wrap env into VecEnv
    else:
        print(f"Using {args.n_envs} parallel envts")
        env = make_vec_env("L5-CLE-v0", env_kwargs={'sim_cfg': SimulationConfigGym(args.eps_length)},
                           n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
                           vec_env_kwargs=dict(start_method='fork'))

    # Custom Feature Extractor backbone
    policy_kwargs = dict(
        features_extractor_class=feature_extractor.SimpleCNN,  # ResNetCNN
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False
    )

    # define model
    print("Creating Model.....")
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.num_rollout_steps,
                learning_rate=get_linear_fn(3e-4, 3e-6, 0.5), gamma=args.gamma, tensorboard_log=args.tb_log, n_epochs=args.n_epochs,
                device=args.device, clip_range=get_linear_fn(0.2, 0.001, 1.0))

    # make eval env at start itself
    print("Creating Eval env.....")
    eval_env = gym.make("L5-CLE-v0", sim_cfg=SimulationConfigGym(args.eps_length))
    model.eval_env = eval_env

    # init callback list
    callback = get_callback_list(args.output_prefix, args.n_envs, args.save_freq, args)

    # train
    start = time.time()
    model.learn(args.n_steps, callback=callback)
    print("Training Time: ", time.time() - start)

    # Eval at end of training
    if args.eval:
        print("Deterministic=True Eval")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        print("Done")

        print("Deterministic=False Eval")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        print("Done")
