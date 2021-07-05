import random
from collections import defaultdict
from typing import List, Dict, Optional

import gym
import numpy as np
from gym import spaces
import torch
from torch.utils.data.dataloader import default_collate

from l5kit.data import AGENT_DTYPE, PERCEPTION_LABEL_TO_INDEX
from l5kit.dataset import EgoDataset
from l5kit.rasterization import Rasterizer
from l5kit.dataset.utils import move_to_device, move_to_numpy
from l5kit.geometry import rotation33_as_yaw, transform_points
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import ClosedLoopSimulator, UnrollInputOutput, SimulationOutput
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator
from l5kit.environment.cle_utils import SimulationOutputGym
from l5kit.environment.utils import default_collate_numpy

    
# Toy environments for benchmarking rasterization process
class L5RasterBaseEnv(gym.Env):
    """
    Custom Environment of L5 Kit that follows gym interface.
    This is a simple env where a random frame is sampled as first observation.
    Frames are then outputted sequentially based on the sampled frame.
    Action has no effect on the next frame.
    """

    def __init__(self, dataset: EgoDataset, rasterizer: Rasterizer, num_simulation_steps: int) -> None:
        """
        :param dataset: The dataset to sample scenes to be rolled out
        :param rasterizer: The dataset rasterizer to generate observations
        :param num_simulation_steps: the number of steps to rollout per scene
        """
        super(L5RasterBaseEnv, self).__init__()
        print("Initializing Base Environment")
        self.dataset = dataset
        n_channels = rasterizer.num_channels()
        raster_size = rasterizer.raster_size[0]
        self.num_simulation_steps = num_simulation_steps

        # Define action and observation space
        # Dummy Action Space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # Observation Space (Raster Image)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(n_channels, raster_size, raster_size), dtype=np.float32)

    def reset(self, max_frame_id: int = 99) -> gym.spaces.Box:
        """
        :param max_frame_id: the maximum frame index to sample from dataset
        :return: the observation (raster image) of frame index sampled
        """
        self.frame_id = random.randint(0, max_frame_id)
        # In L5Kit, each scene usually consists of 248 frames
        self.max_frame_id = self.frame_id + self.num_simulation_steps

        obs = self.dataset[self.frame_id]["image"]
        return obs

    def step(self, action: gym.spaces.Discrete) -> gym.spaces.Box:
        """
        :param action: dummy action. Next frame is returned.
        :return: the observation of next frame
        """
        self.frame_id += 1
        obs = self.dataset[self.frame_id]["image"]

        # done is True when episode ends
        done = (self.frame_id == self.max_frame_id)

        # reward always set to 1
        reward = 1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass