import time
import argparse

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback

from l5kit.environment.feature_extractor import ResNetCNN, SimpleCNN
from l5kit.environment.callbacks import VizCallback, TrajectoryCallback

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_subProc', action='store_true',
                        help='Use SubProc environment')
    parser.add_argument('--n_steps', default=3000, type=int,
                        help='Number of time steps')
    parser.add_argument('--num_rollout_steps', default=64, type=int,
                        help='Number of rollout steps per update')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Discount factor')
    parser.add_argument('--tb_log', default=None, type=str,
                        help='Tensorboard Log folder')
    parser.add_argument('--save', action='store_true',
                        help='Save model')
    parser.add_argument('--load', action='store_true',
                        help='load TRAINED model')
    parser.add_argument('--eval', action='store_true',
                        help='Eval model at end of training')
    parser.add_argument('--clt', action='store_true',
                        help='Closed Loop training')
    parser.add_argument('--ckpt_prefix', default=None, type=str,
                        help='Ckpt prefix for saving model')
    parser.add_argument('--output_prefix', required=True, type=str,
                        help='Output prefix for tracking model outputs during training')
    args = parser.parse_args()

    # make gym env
    if not args.use_subProc:
        env = gym.make("L5-v0")
        env.clt = args.clt
    # custom wrap env into VecEnv
    else:
        env = make_vec_env("L5-v0", n_envs=2, vec_env_cls=SubprocVecEnv,
                           vec_env_kwargs=dict(start_method='fork'))
        env.set_attr("clt", args.clt)

    # Custom Feature Extractor backbone
    policy_kwargs = dict(
        features_extractor_class=SimpleCNN, #ResNetCNN
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False
    )

    # define model
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.num_rollout_steps,
                learning_rate=3e-4, gamma=args.gamma, tensorboard_log=args.tb_log)
    print("Model created.....")

    # make eval env at start itself
    print("Creating Eval env.....")
    eval_env = gym.make("L5-v0")
    # Logs will be saved in log_dir/monitor.csv
    eval_env = Monitor(eval_env, './monitor')
    eval_env.clt = args.clt
    model.eval_env = eval_env

    # Define callbacks
    assert args.output_prefix is not None
    viz_callback = VizCallback(save_freq=10000, save_path='./logs/', 
                               name_prefix=args.output_prefix, verbose=2)
    callback = CallbackList([viz_callback])
    if args.save:
        ckpt_prefix = args.ckpt_prefix if args.ckpt_prefix is not None else args.output_prefix
        print("Saving Model every 10000 steps")
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                                 name_prefix=ckpt_prefix, verbose=2)
        callback = CallbackList([viz_callback, checkpoint_callback])
    # traj_callback = TrajectoryCallback(save_path='./logs/', verbose=2)


    # train only when not loading saved model
    if not args.load:
        start = time.time()
        model.learn(args.n_steps, callback=callback)
        print("Training Time: ", time.time() - start)

    # Load saved model
    if args.load:
        assert args.save is False
        print("Loading trained model..........")
        model.load("saved_models/PPO_OpenLoop")

    # Eval
    if args.eval:
        print("Deterministic=True Eval")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        print("Done")

        print("Deterministic=False Eval")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        print("Done")