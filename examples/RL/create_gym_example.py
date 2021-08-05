import os

import gym
import l5kit.environment


# TODO: Create a Google colab notebook for environment creation and training.

# By setting the L5KIT_DATA_FOLDER variable, we can point the script
# to the folder where the data lies.
os.environ["L5KIT_DATA_FOLDER"] = os.environ["HOME"] + '/level5_data'
env_config_path = "./config.yaml"

# call
env = gym.make('L5-CLE-v0', env_config_path=env_config_path)

# get initial observation on reset
obs = env.reset()
print("Observation Shape: ", obs["image"].shape)
