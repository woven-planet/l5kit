import argparse
import time
from typing import Optional

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.environment.callbacks import TrajectoryCallback, VizCallback
from l5kit.environment.feature_extractor import ResNetCNN, SimpleCNN


def get_callback_list(output_prefix: str, n_envs: int, clt: bool, save_freq: Optional[int] = 50000,
                      ckpt_prefix: Optional[str] = None) -> CallbackList:

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

    # Print Error at end of OpenLoop training
    if not clt:
        traj_callback = TrajectoryCallback(save_path='./logs/', verbose=2)
        callback_list.append(traj_callback)

    callback = CallbackList(callback_list)
    return callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_prefix', required=True, type=str,
                        help='Output prefix for tracking model outputs during training')
    parser.add_argument('--save_freq', default=50000, type=int,
                        help='Frequency to save model states')
    parser.add_argument('--n_envs', default=4, type=int,
                        help='Number of parallel envts')
    parser.add_argument('--n_steps', default=3000, type=int,
                        help='Number of time steps')
    parser.add_argument('--num_rollout_steps', default=64, type=int,
                        help='Number of rollout steps per update')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Discount factor')
    parser.add_argument('--tb_log', default=None, type=str,
                        help='Tensorboard Log folder')
    parser.add_argument('--eval', action='store_true',
                        help='Eval model at end of training')
    parser.add_argument('--clt', action='store_true',
                        help='Closed Loop training')
    parser.add_argument('--ckpt_prefix', default=None, type=str,
                        help='Ckpt prefix for saving model')
    args = parser.parse_args()

    # make gym env
    if args.n_envs == 1:
        print("Using 1 envt")
        env = gym.make("L5-v0")
        env.clt = args.clt
    # custom wrap env into VecEnv
    else:
        print(f"Using {args.n_envs} parallel envts")
        env = make_vec_env("L5-v0", n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
                           vec_env_kwargs=dict(start_method='fork'))
        env.env_method("set_clt", args.clt)

    # Custom Feature Extractor backbone
    policy_kwargs = dict(
        features_extractor_class=SimpleCNN,  # ResNetCNN
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False
    )

    # define model
    print("Creating Model.....")
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.num_rollout_steps,
                learning_rate=3e-4, gamma=args.gamma, tensorboard_log=args.tb_log)

    # make eval env at start itself
    print("Creating Eval env.....")
    eval_env = gym.make("L5-v0")
    eval_env.clt = args.clt
    model.eval_env = eval_env

    # init callback list
    callback = get_callback_list(args.output_prefix, args.n_envs, args.clt, args.save_freq)

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
