import argparse
import os

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.environment.feature_extractor import CustomFeatureExtractor

# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
os.environ["L5KIT_DATA_FOLDER"] = os.environ["HOME"] + '/level5_data/'
if "L5KIT_DATA_FOLDER" not in os.environ:
    raise KeyError("L5KIT_DATA_FOLDER environment variable not set")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./gym_config.yaml',
                        help='Path to L5Kit environment config file')
    parser.add_argument('-o', '--output', type=str, default='PPO',
                        help='File name for saving model states')
    parser.add_argument('--load', type=str,
                        help='Path to load model and continue training')
    parser.add_argument('--tb_log', default=None, type=str,
                        help='Tensorboard log folder')
    parser.add_argument('--save_path', default='./logs/', type=str,
                        help='Folder to save model checkpoints')
    parser.add_argument('--save_freq', default=1000, type=int,
                        help='Frequency to save model checkpoints')
    parser.add_argument('--eval_freq', default=1000, type=int,
                        help='Frequency to evaluate model state')
    parser.add_argument('--n_eval_episodes', default=10, type=int,
                        help='Number of episodes to evaluate')
    parser.add_argument('--n_steps', default=1000000, type=int,
                        help='Total number of training time steps')
    parser.add_argument('--train_freq', default=1, type=int,
                        help='Update the model every ``train_freq`` steps')
    parser.add_argument('--gradient_steps', default=1, type=int,
                        help='Number of gradient_steps per update')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Mini batch size of model update')
    parser.add_argument('--buffer_size', default=30000, type=float,
                        help='Buffer Size')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', default=0.90, type=float,
                        help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator')
    parser.add_argument('--model_arch', default='simple_gn', type=str,
                        help='Model architecture of feature extractor')
    parser.add_argument('--features_dim', default=128, type=int,
                        help='Output dimension of feature extractor')
    parser.add_argument('--eps_length', default=32, type=int,
                        help='Episode length of gym rollouts')
    parser.add_argument('--rew_clip', default=15, type=float,
                        help='Reward clipping threshold')
    parser.add_argument('--kinematic', action='store_true',
                        help='Flag to use kinematic model in the environment')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # make train env
    env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic}
    env = make_vec_env("L5-CLE-v0", env_kwargs=env_kwargs)

    # Custom Feature Extractor backbone
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": args.features_dim, "model_arch": args.model_arch},
        "normalize_images": False
    }

    # define model
    if args.load is not None:
        model = PPO.load(args.load, env)
    else:
        model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, train_freq=args.train_freq,
                    learning_rate=args.lr, gamma=args.gamma, tensorboard_log=args.tb_log, gradient_steps=args.gradient_steps,
                    buffer_size=args.buffer_size)

    # make eval env
    eval_env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'return_info': True}
    eval_env = make_vec_env("L5-CLE-v0", env_kwargs=eval_env_kwargs)

    # callbacks
    # Save Model Periodically
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.save_path,
                                             name_prefix=args.output)

    # Eval Model Periodically
    eval_callback = EvalCallback(eval_env, eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes)

    # train
    model.learn(args.n_steps, callback=[checkpoint_callback, eval_callback])
