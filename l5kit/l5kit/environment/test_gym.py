import os
import argparse

import gym

from l5kit.environment.register_l5_env import create_l5_env

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/ubuntu/level5_data/',
                    help='Environment configuration file')
parser.add_argument('--env_config_path', type=str, default='/home/ubuntu/src/l5kit/examples/RL/config.yaml',
                    help='Environment configuration file')
parser.add_argument('--eps_length', default=16, type=int,
                    help='Number of time steps')
parser.add_argument('--rew_clip', default=15, type=float,
                    help='Reward Clipping Threshold')
args = parser.parse_args()

# By setting the L5KIT_DATA_FOLDER variable, we can point the script to the folder where the data lies.
# The path to 'l5_data_path' needs to be provided
os.environ["L5KIT_DATA_FOLDER"] = args.data_path

# create and register L5Kit gym env
create_l5_env(args)

# call
env = gym.make('L5-CLE-v0')
