import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.data import get_dataset_path
from l5kit.environment.callbacks import L5KitEvalCallback
from l5kit.environment.envs.l5_env import SimulationConfigGym
from l5kit.environment.feature_extractor import CustomFeatureExtractor

# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
os.environ["L5KIT_DATA_FOLDER"], _ = get_dataset_path()
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
    parser.add_argument('--simnet', action='store_true',
                        help='Use simnet to control agents')
    parser.add_argument('--simnet_model_path', default=None, type=str,
                        help='Path to simnet model that controls agents')
    parser.add_argument('--tb_log', default=None, type=str,
                        help='Tensorboard log folder')
    parser.add_argument('--save_path', default='./logs/', type=str,
                        help='Folder to save model checkpoints')
    parser.add_argument('--save_freq', default=100000, type=int,
                        help='Frequency to save model checkpoints')
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
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=0.95, type=float,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', default=0.90, type=float,
                        help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator')
    parser.add_argument('--clip_start_val', default=0.1, type=float,
                        help='Start value of clipping in PPO')
    parser.add_argument('--clip_end_val', default=0.001, type=float,
                        help='End value of clipping in PPO')
    parser.add_argument('--clip_progress_ratio', default=1.0, type=float,
                        help='Training progress ratio to end linear schedule of clipping')
    parser.add_argument('--model_arch', default='simple_gn', type=str,
                        help='Model architecture of feature extractor')
    parser.add_argument('--features_dim', default=128, type=int,
                        help='Output dimension of feature extractor')
    parser.add_argument('--n_envs', default=4, type=int,
                        help='Number of parallel environments')
    parser.add_argument('--n_eval_envs', default=4, type=int,
                        help='Number of parallel environments for evaluation')
    parser.add_argument('--eps_length', default=32, type=int,
                        help='Episode length of gym rollouts')
    parser.add_argument('--rew_clip', default=15, type=float,
                        help='Reward clipping threshold')
    parser.add_argument('--kinematic', action='store_true',
                        help='Flag to use kinematic model in the environment')
    parser.add_argument('--enable_scene_type_aggregation', action='store_true',
                        help='enable scene type aggregation of evaluation metrics')
    parser.add_argument('--scene_id_to_type_path', default=None, type=str,
                        help='Path to csv file mapping scene id to scene type')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # Simnet model
    if args.simnet and (args.simnet_model_path is None):
        raise ValueError("simnet_model_path needs to be provided when using simnet")

    # make train env
    train_sim_cfg = SimulationConfigGym()
    train_sim_cfg.num_simulation_steps = args.eps_length + 1
    train_sim_cfg.use_agents_gt = (not args.simnet)
    env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'train': True,
                  'sim_cfg': train_sim_cfg, 'simnet_model_path': args.simnet_model_path}
    env = make_vec_env("L5-CLE-v0", env_kwargs=env_kwargs, n_envs=args.n_envs,
                       vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

    # Custom Feature Extractor backbone
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": args.features_dim, "model_arch": args.model_arch},
        "normalize_images": False
    }

    # define model
    clip_schedule = get_linear_fn(args.clip_start_val, args.clip_end_val, args.clip_progress_ratio)
    if args.load is not None:
        model = PPO.load(args.load, env, clip_range=clip_schedule, learning_rate=args.lr)
    else:
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.num_rollout_steps,
                    learning_rate=args.lr, gamma=args.gamma, tensorboard_log=args.tb_log, n_epochs=args.n_epochs,
                    clip_range=clip_schedule, batch_size=args.batch_size, seed=args.seed, gae_lambda=args.gae_lambda)

    # make eval env
    eval_sim_cfg = SimulationConfigGym()
    eval_sim_cfg.num_simulation_steps = None
    eval_sim_cfg.use_agents_gt = (not args.simnet)
    eval_env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'return_info': True,
                       'train': False, 'sim_cfg': eval_sim_cfg, 'simnet_model_path': args.simnet_model_path}
    eval_env = make_vec_env("L5-CLE-v0", env_kwargs=eval_env_kwargs, n_envs=args.n_eval_envs,
                            vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

    # callbacks
    # Note: When using multiple environments, each call to ``env.step()``
    # will effectively correspond to ``n_envs`` steps.
    # To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    # Save Model Periodically
    checkpoint_callback = CheckpointCallback(save_freq=(args.save_freq // args.n_envs), save_path=args.save_path,
                                             name_prefix=args.output)

    # Eval Model Periodically
    eval_callback = L5KitEvalCallback(eval_env, eval_freq=(args.eval_freq // args.n_envs),
                                      n_eval_episodes=args.n_eval_episodes, n_eval_envs=args.n_eval_envs,
                                      prefix='l5_cle_eval', enable_scene_type_aggregation=args.enable_scene_type_aggregation,
                                      scene_id_to_type_path=args.scene_id_to_type_path)

    # train
    model.learn(args.n_steps, callback=[checkpoint_callback, eval_callback])
