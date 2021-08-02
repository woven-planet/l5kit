import os
import random
from typing import Any, Dict, Optional

import gym
import numpy as np
from gym import spaces

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer

from .l5_env import GymStepOutput


# Toy environments for benchmarking rasterization process
class L5RasterBaseEnvV1(gym.Env):
    """
    Custom Environment of L5 Kit that follows gym interface.
    This is a simple env where a random frame is sampled as first observation.
    Frames are then outputted sequentially based on the sampled frame.
    Action has no effect on the next frame.
    """

    def __init__(self) -> None:
        super(L5RasterBaseEnvV1, self).__init__()
        print("Initializing Base V1 Environment")
        # set env variable for data
        os.environ["L5KIT_DATA_FOLDER"] = "/home/ubuntu/level5_data/"

        dm = LocalDataManager(None)
        # get config
        cfg = load_config_data("./envs/config.yaml")

        # rasterisation
        rasterizer = build_rasterizer(cfg, dm)
        raster_size = cfg["raster_params"]["raster_size"][0]
        print("Raster Size: ", raster_size)

        # init dataset
        train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
        self.dataset = EgoDataset(cfg, train_zarr, rasterizer)

        # init gym environment
        self.num_simulation_steps = cfg["gym_params"]["num_simulation_steps"]

        n_channels = rasterizer.num_channels()

        # Define action and observation space
        # Dummy Action Space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # Observation Space: gym.spaces.Dict (image: [n_channels, raster_size, raster_size])
        self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=1,
                                              shape=(n_channels, raster_size, raster_size), dtype=np.float32)})

    def reset(self, max_frame_id: int = 99) -> Dict[str, np.ndarray]:
        """
        :param max_frame_id: the maximum frame index to sample from dataset
        :return: the observation (raster image) of frame index sampled
        """
        self.frame_id = random.randint(0, max_frame_id)
        # In L5Kit, each scene usually consists of 248 frames
        self.max_frame_id = self.frame_id + self.num_simulation_steps

        obs = {"image": self.dataset[self.frame_id]["image"]}
        return obs

    def step(self, action: np.ndarray) -> GymStepOutput:
        """
        :param action: dummy action. Next frame is returned.
        :return: the observation of next frame
        """
        self.frame_id += 1
        obs = {"image": self.dataset[self.frame_id]["image"]}

        # done is True when episode ends
        done = (self.frame_id == self.max_frame_id)

        # reward always set to 1
        reward = 1

        # Optionally we can pass additional info, we are not using that for now
        info: Dict[str, Any] = {}

        return GymStepOutput(obs, reward, done, info)
        # return obs, reward, done, info

    def render(self, mode: Optional[str] = 'console') -> None:
        pass

    def close(self) -> None:
        pass
