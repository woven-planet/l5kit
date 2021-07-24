import os

import gym
import l5kit.environment

# By setting the L5KIT_DATA_FOLDER variable, we can point the script
# to the folder where the data lies.
os.environ["L5KIT_DATA_FOLDER"] = os.environ["HOME"] + '/level5_data'
env_config_path = "./config.yaml"

# call
env = gym.make('L5-CLE-v0', env_config_path=env_config_path)
