import os

import gym
import l5kit.environment


# TODO: Create a Google colab notebook for environment creation and training.

# Before starting, please download the Lyft L5 Prediction Dataset 2020 and
# follow the instructions (https://github.com/lyft/l5kit#download-the-datasets)
# to correctly organise it.

# Set the L5KIT_DATA_FOLDER path and uncomment the line below
# os.environ["L5KIT_DATA_FOLDER"] = your_path_to_data
env_config_path = "./config.yaml"

# call
env = gym.make('L5-CLE-v0', env_config_path=env_config_path)

# get initial observation on reset
obs = env.reset()
print("Observation Shape: ", obs["image"].shape)
