import random

import gym
import numpy as np
from gym import spaces

class L5RasterEnv1(gym.Env):
    """
    Custom Environment of L5 Kit that follows gym interface.
    This is a simple env where a random scene is rolled out. 
    Each frame sequentially is returned as observation.
    """

    def __init__(self, input_dataset, sample_function, n_channels, raster_size):
        super(L5RasterEnv1, self).__init__()

        self.input_dataset = input_dataset
        self.sample_function = sample_function
        self.raster_size = raster_size

        # Define action and observation space
        # They must be gym.spaces objects
        # DUMMY Action Space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(n_channels, raster_size, raster_size), dtype=np.float32)

    def reset(self, max_scene_id=999):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Sample a episode randomly
        self.scene_id = random.randint(0, max_scene_id)
        self.episode_scene = self.input_dataset[self.scene_id]
        self.frame_id = 0
        self.max_frame_id = len(self.episode_scene["frames"]) - 1

        # get initial obs
        data = self.sample_function(self.frame_id, self.episode_scene["frames"], \
                                    self.episode_scene["agents"], \
                                    self.episode_scene["tl_faces"], None)
        obs = data["image"].transpose(2, 0, 1) 
        return obs

    def step(self, action):
        self.frame_id += 1
        data = self.sample_function(self.frame_id, self.episode_scene["frames"], \
                                    self.episode_scene["agents"], \
                                    self.episode_scene["tl_faces"], None)
        obs = data["image"].transpose(2, 0, 1) 

        # done is True when episode ends
        done = (self.frame_id == self.max_frame_id)

        # reward always set to 1
        reward = 1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        # return np.array([self.agent_pos]).astype(np.float32), reward, done, info
        return obs, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass


class L5RasterEnv2(gym.Env):
    """
    Custom Environment of L5 Kit that follows gym interface.
    This is a simple env where a random frame is returned as observation.
    """

    def __init__(self, input_dataset, n_channels, raster_size):
        super(L5RasterEnv2, self).__init__()

        self.input_dataset = input_dataset
        self.raster_size = raster_size

        # Define action and observation space
        # They must be gym.spaces objects
        # DUMMY Action Space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(n_channels, raster_size, raster_size), dtype=np.float32)

    def reset(self, max_frame_id=9999):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.frame_id = random.randint(0, max_frame_id)
        # In L5Kit, each scene usually consists of 248 frames
        self.max_frame_id = self.frame_id + 248

        obs = self.input_dataset[self.frame_id]["image"]
        return obs

    def step(self, action):
        self.frame_id += 1
        obs = self.input_dataset[self.frame_id]["image"]

        # done is True when episode ends
        done = (self.frame_id == self.max_frame_id)

        # reward always set to 1
        reward = 1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        # return np.array([self.agent_pos]).astype(np.float32), reward, done, info
        return obs, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass
