import argparse
import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.environment.callbacks import get_callback_list
from l5kit.environment.feature_extractor import CustomFeatureExtractor
from l5kit.environment.monitor_utils import monitor_env


os.environ["L5KIT_DATA_FOLDER"] = os.environ["HOME"] + '/level5_data/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../../examples/RL/config.yaml',
                        help='Path to L5Kit environment config file')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Output file for saving model states')
    parser.add_argument('--load', type=str,
                        help='Path to load model and continue training')
    parser.add_argument('--tb_log', default=None, type=str,
                        help='Tensorboard log folder')
    parser.add_argument('--ckpt_prefix', default=None, type=str,
                        help='Ckpt prefix for saving model')
    parser.add_argument('--save_freq', default=100000, type=int,
                        help='Frequency to save model state')
    parser.add_argument('--eval_freq', default=100000, type=int,
                        help='Frequency to evaluate model state')
    parser.add_argument('--n_eval_episodes', default=10, type=int,
                        help='Number of episodes to evaluate')
    parser.add_argument('--n_steps', default=1000000, type=int,
                        help='Total number of training time steps')
    parser.add_argument('--num_rollout_steps', default=256, type=int,
                        help='Number of rollout steps per environment per model update')
    parser.add_argument('--n_epochs', default=10, type=int,
                        help='Number of model training epochs per update')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini batch size of model update')
    parser.add_argument('--gamma', default=0.95, type=float,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', default=0.95, type=float,
                        help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator')
    parser.add_argument('--clip_start_val', default=0.2, type=float,
                        help='Start value of clipping in PPO')
    parser.add_argument('--clip_end_val', default=0.001, type=float,
                        help='End value of clipping in PPO')
    parser.add_argument('--clip_progress_ratio', default=1.0, type=float,
                        help='Training progress ratio to end linear schedule of clipping')
    parser.add_argument('--model_arch', default='simple', type=str,
                        help='Model architecture of feature extractor')
    parser.add_argument('--n_envs', default=4, type=int,
                        help='Number of parallel environments')
    parser.add_argument('--eps_length', default=32, type=int,
                        help='Episode length of gym rollouts')
    parser.add_argument('--rew_clip', default=15, type=float,
                        help='Reward clipping threshold')
    parser.add_argument('--kinematic', action='store_true',
                        help='Flag to use kinematic model in the environment')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # Extra info keywords to monitor in addition to tensorboard logs
    info_keywords = ("reward_tot", "reward_dist", "reward_yaw")
    monitor_dir = 'monitor_logs/{}'.format(args.output)
    monitor_kwargs = {'info_keywords': info_keywords}

    # make train env
    if args.n_envs == 1:
        env = gym.make("L5-CLE-v0", env_config_path=args.config, use_kinematic=args.kinematic)
        env = monitor_env(env, monitor_dir, monitor_kwargs)
    else:
        env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'train': True}
        env = make_vec_env("L5-CLE-v0", env_kwargs=env_kwargs, n_envs=args.n_envs,
                           vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'),
                           monitor_dir=monitor_dir, monitor_kwargs=monitor_kwargs)

    # Custom Feature Extractor backbone
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128, model_arch=args.model_arch),
        normalize_images=False
    )

    # define model
    clip_schedule = get_linear_fn(args.clip_start_val, args.clip_end_val, args.clip_progress_ratio)
    if args.load is not None:
        print("Loading Model......")
        model = PPO.load(args.load, env, clip_range=clip_schedule)
    else:
        print("Creating Model.....")
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.num_rollout_steps,
                    learning_rate=3e-4, gamma=args.gamma, tensorboard_log=args.tb_log, n_epochs=args.n_epochs,
                    clip_range=clip_schedule, batch_size=args.batch_size, seed=args.seed, gae_lambda=args.gae_lambda)

    # make eval env
    eval_env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'train': False,
                       'return_info': True}
    eval_env = make_vec_env("L5-CLE-v0", env_kwargs=eval_env_kwargs, n_envs=4,
                            vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))

    # init callback list
    callback = get_callback_list(eval_env, args.output, args.n_envs, args.save_freq)

    # train
    model.learn(args.n_steps, callback=callback)
