import argparse
import os
import time

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.environment.callbacks import get_callback_list
from l5kit.environment.envs.l5_env import SimulationConfigGym
from l5kit.environment.feature_extractor import CustomFeatureExtractor
from l5kit.environment.monitor_utils import monitor_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_prefix', required=True, type=str,
                        help='Output prefix for tracking model outputs during training')
    parser.add_argument('--data_path', type=str, default='/level5_data/',
                        help='Environment configuration file')
    parser.add_argument('--disable_cle', action='store_true',
                        help='Flag to disable close loop environment')
    parser.add_argument('--save_freq', default=50000, type=int,
                        help='Frequency to save model states')
    parser.add_argument('--n_envs', default=4, type=int,
                        help='Number of parallel envts')
    parser.add_argument('--eps_length', default=32, type=int,
                        help='Number of time steps')
    parser.add_argument('--n_steps', default=1000000, type=int,
                        help='Number of time steps')
    parser.add_argument('--num_rollout_steps', default=256, type=int,
                        help='Number of rollout steps per update')
    parser.add_argument('--n_epochs', default=10, type=int,
                        help='Number of model epochs per update')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini batch size of model update')
    parser.add_argument('--gamma', default=0.95, type=float,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', default=0.95, type=float,
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
    parser.add_argument('--kinematic', action='store_true',
                        help='Use kinematic model')
    parser.add_argument('--clip_start_val', default=0.2, type=float,
                        help='Start value of clip in PPO')
    parser.add_argument('--clip_end_val', default=0.001, type=float,
                        help='End value of clip in PPO')
    parser.add_argument('--model_arch', default='simple', type=str,
                        help='Model architecture of feature extractor')
    parser.add_argument('--load_model_path', type=str,
                        help='Path to load model and continue training')
    parser.add_argument('--clip_range_vf', default=None, type=float,
                        help='Clip value of value function in PPO')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # By setting the L5KIT_DATA_FOLDER variable, we can point the script to the folder where the data lies.
    # The path to 'l5_data_path' needs to be provided
    os.environ["L5KIT_DATA_FOLDER"] = os.environ["HOME"] + args.data_path

    # Extra info keywords to monitor in addition to tensorboard logs
    info_keywords = ("reward_tot", "reward_dist", "reward_yaw")
    monitor_dir = 'monitor_logs/{}'.format(args.output_prefix)
    monitor_kwargs = {'info_keywords': info_keywords}

    # make gym env
    if args.n_envs == 1:
        print("Using 1 envt")
        env = gym.make("L5-CLE-v0", sim_cfg=SimulationConfigGym(args.eps_length), use_kinematic=args.kinematic)
        env = monitor_env(env, monitor_dir, monitor_kwargs)

    # custom wrap env into VecEnv
    else:
        print(f"Using {args.n_envs} parallel envts")
        env_kwargs = {'sim_cfg': SimulationConfigGym(args.eps_length), 'use_kinematic': args.kinematic}
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
    clip_schedule = get_linear_fn(args.clip_start_val, args.clip_end_val, 1)
    if args.load_model_path is not None:
        print("Loading Model......")
        model = PPO.load(args.load_model_path, env, clip_range=clip_schedule)
        # _ = model.env.reset()  # Else, error thrown to reset the environment
        # reset_num_timesteps = False
        reset_num_timesteps = True
    else:
        print("Creating Model.....")
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.num_rollout_steps,
                    learning_rate=3e-4, gamma=args.gamma, tensorboard_log=args.tb_log, n_epochs=args.n_epochs,
                    device=args.device, clip_range=clip_schedule, clip_range_vf=args.clip_range_vf,
                    batch_size=args.batch_size, seed=args.seed, gae_lambda=args.gae_lambda)
        reset_num_timesteps = True

    # make eval env at start itself
    print("Creating Eval env.....")
    eval_env = gym.make("L5-CLE-v0", sim_cfg=SimulationConfigGym(args.eps_length), use_kinematic=args.kinematic,
                        return_info=True)
    model.eval_env = eval_env

    # init callback list
    callback = get_callback_list(args.output_prefix, args.n_envs, args.save_freq)

    # train
    start = time.time()
    model.learn(args.n_steps, callback=callback, reset_num_timesteps=reset_num_timesteps)
    print("Training Time: ", time.time() - start)
