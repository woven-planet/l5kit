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
from .utils import default_collate_numpy

class L5Env(gym.Env):
    """
    Custom Gym Environment of L5 Kit.
    This is a full implemented env where a random scene is rolled out. 
    Each frame sequentially is returned as observation.
    An action is taken (dummy/predicted). 
    The raster is updated according to predicted ego positions
    """

    def __init__(self, dataset: EgoDataset, rasterizer: Rasterizer, future_num_frames: int,
                 num_simulation_steps: int, det_roll: bool) -> None:
        """
        :param dataset: The dataset to sample scenes to be rolled out
        :param rasterizer: The dataset rasterizer to generate observations
        :param future_num_frames: the number of frames to predict
        :param num_simulation_steps: the number of steps to rollout per scene
        :param det_roll: flag for deterministic rollout, i.e., select the scenes
                         in a deterministic manner (sequentially from 0)
        """
        super(L5Env, self).__init__()
        print("Initializing L5Kit Custom Gym Environment")

        self.dataset = dataset
        # Define action and observation space
        # Continuous Action Space: gym.spaces.Dict (X, Y, Yaw * number of future states)
        self.action_space = spaces.Dict({
                                'positions': spaces.Box(low=-1000, high=1000, shape=(1, future_num_frames, 2)),
                                'yaws': spaces.Box(low=-1000, high=1000, shape=(1, future_num_frames, 1)),
                                'velocities': spaces.Box(low=-1000, high=1000, shape=(1, future_num_frames, 2))})

        # Observation Space: gym.spaces.Dict (image: [n_channels, raster_size, raster_size])
        n_channels = rasterizer.num_channels()
        raster_size = rasterizer.raster_size[0]
        self.observation_space = spaces.Dict({
                                    'image': spaces.Box(low=0, high=1,
                                    shape=(n_channels, raster_size, raster_size), dtype=np.float32)})

        # Define Close-Loop Simulator
        self.sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                                        distance_th_far=30, distance_th_close=15,
                                        num_simulation_steps=num_simulation_steps,
                                        start_frame_index=0, show_info=True)
        
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.simulator = ClosedLoopSimulator(self.sim_cfg, self.dataset, self.device)

        # deterministic rollout
        self.det_roll = det_roll
        self.det_scene_idx = 0

    def reset(self, max_scene_id: int = 99) -> gym.spaces.Dict:
        """
        :param max_scene_id: the maximum scene index to sample from dataset
        :return: the observation of first frame of sampled scene index
        """
        if self.det_roll:
            # Deterministic episode selection
            self.scene_indices = [self.det_scene_idx]
            self.det_scene_idx += 1
        else:
            # Sample a episode randomly
            self.scene_indices = [random.randint(0, max_scene_id)]
        self.sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, self.scene_indices, self.sim_cfg)

        # Define in / outs for given scene
        self.agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        self.ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        # Output first observation
        self.frame_index = 0
        ego_input = self.sim_dataset.rasterise_frame_batch(self.frame_index)
        self.ego_input_dict = default_collate_numpy(ego_input[0])

        # Only output the image attribute
        obs = {'image': ego_input[0]["image"]}
        # print("FI: ", self.frame_index, "SI: ", self.scene_indices)
        return obs


    def step(self, action: gym.spaces.Dict) -> gym.spaces.Dict:
        """
        Analogous to sim_loop.unroll(scenes_to_unroll) in L5Kit

        :param action: the action to perform on current state/frame
        :return: the observation of next frame based on current action
        """

        frame_index = self.frame_index
        next_frame_index = frame_index + 1
        should_update = next_frame_index != len(self.sim_dataset)

        # EGO
        if not self.sim_cfg.use_ego_gt:
            self.ego_output_dict = action

            if should_update:
                self.simulator.update_ego(self.sim_dataset, next_frame_index, self.ego_input_dict, self.ego_output_dict)

            ego_frame_in_out = self.simulator.get_ego_in_out(self.ego_input_dict, self.ego_output_dict, self.simulator.keys_to_exclude)
            for scene_idx in self.scene_indices:
                self.ego_ins_outs[scene_idx].append(ego_frame_in_out[scene_idx])

        # Prepare next obs
        if should_update:
            self.frame_index += 1
            ego_input = self.sim_dataset.rasterise_frame_batch(self.frame_index)
            self.ego_input_dict = default_collate_numpy(ego_input[0])
            obs = {"image": ego_input[0]["image"]}
            # print("FI: ", self.frame_index, "SI: ", self.scene_indices, "Update: ", should_update)
        else:
            # Dummy final obs (when done is True)
            ego_input = self.sim_dataset.rasterise_frame_batch(0)
            obs = {"image": ego_input[0]["image"]}

        # done is True when episode ends
        done = not should_update
        if done:
            # generate simulated_outputs
            simulated_outputs: List[SimulationOutput] = []
            for scene_idx in self.scene_indices:
                simulated_outputs.append(SimulationOutput(scene_idx, self.sim_dataset, self.ego_ins_outs, self.agents_ins_outs))

        # reward set to 1 when done
        reward = 0
        if done:
            reward = 1

        # Optionally we can pass additional info, we are using that to output simulated outputs
        info = {}
        if done:
            info = {"info": simulated_outputs}

        return obs, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass


# Toy environments for benchmarking rasterization process
class L5RasterBaseEnv(gym.Env):
    """
    Custom Environment of L5 Kit that follows gym interface.
    This is a simple env where a random frame is sampled as first observation.
    Frames are then outputted sequentially based on the sampled frame.
    Action has no effect on the next frame.
    """

    def __init__(self, dataset: EgoDataset, rasterizer: Rasterizer) -> None:
        """
        :param dataset: The dataset to sample scenes to be rolled out
        :param rasterizer: The dataset rasterizer to generate observations
        """
        super(L5RasterBaseEnv, self).__init__()
        print("Initializing Base Environment")
        self.dataset = dataset
        n_channels = rasterizer.num_channels()
        raster_size = rasterizer.raster_size[0]

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
        self.max_frame_id = self.frame_id + 248

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

class L5RasterCacheEnv(gym.Env):
    """
    Toy Custom Environment of L5 Kit that follows gym interface.
    This is a toy env where a random scene is rolled out.
    Each frame sequentially is returned as observation.
    Action has no effect on the next frame.
    The raster of each frame is pre-computed and cached in the memory.
    """

    def __init__(self, dataset: List[Dict[str, np.ndarray]], rasterizer: Rasterizer) -> None:
        """
        :param dataset: The dataset with pre-computed rasters for each frame
        :param rasterizer: The dataset rasterizer to generate observations
        """
        super(L5RasterCacheEnv, self).__init__()
        print("Initializing Environment with cached raster images")

        self.dataset = dataset
        n_channels = rasterizer.num_channels()
        raster_size = rasterizer.raster_size[0]

        # Define action and observation space
        # Dummy Action Space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # The observation space
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(n_channels, raster_size, raster_size), dtype=np.float32)

    def reset(self, max_scene_id: int = 99) -> gym.spaces.Box:
        """
        :param max_scene_id: the maximum scene index to sample from dataset
        :return: the observation (the pre-computed raster) of frame index sampled
        """
        # Sample a episode randomly
        self.scene_id = random.randint(0, max_scene_id)
        self.episode_scene = self.dataset[self.scene_id]
        self.frame_id = 0
        self.max_frame_id = len(self.episode_scene["frames"]) - 1

        # get initial obs
        obs = self.dataset[self.scene_id]["rasters"][self.frame_id]
        return obs

    def step(self, action: gym.spaces.Discrete) -> gym.spaces.Box:
        """
        :param action: dummy action. Next frame is returned.
        :return: observation (the pre-computed raster) of next frame
        """
        self.frame_id += 1
        obs = self.dataset[self.scene_id]["rasters"][self.frame_id]

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


class L5DatasetCacheEnv(gym.Env):
    """
    Toy Custom Environment of L5 Kit that follows gym interface.
    This is a toy env where a random scene is rolled out.
    Each frame sequentially is returned as observation.
    Action has no effect on the next frame.
    The dataset has been cached in the memory.
    """

    def __init__(self, dataset: List[Dict[str, np.ndarray]], rasterizer: Rasterizer) -> None:
        super(L5DatasetCacheEnv, self).__init__()
        print("Initializing Environment with cached dataset")

        self.dataset = dataset
        n_channels = rasterizer.num_channels()
        raster_size = rasterizer.raster_size[0]

        # Define action and observation space
        # Dummy Action Space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # The observation space
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(n_channels, raster_size, raster_size), dtype=np.float32)

    def reset(self, max_scene_id: int = 99) -> gym.spaces.Box:
        """
        :param max_scene_id: the maximum scene index to sample from dataset
        :return: the observation (raster image) of frame index sampled
        """
        # Sample a episode randomly
        self.scene_id = random.randint(0, max_scene_id)
        self.episode_scene = self.dataset[self.scene_id]
        self.frame_id = 0
        self.max_frame_id = len(self.episode_scene["frames"]) - 1

        # get initial obs
        data = self.sample_function(self.frame_id, self.episode_scene["frames"], \
                                    self.episode_scene["agents"], \
                                    self.episode_scene["tl_faces"], None)
        obs = data["image"].transpose(2, 0, 1)
        return obs

    def step(self, action: gym.spaces.Discrete) -> gym.spaces.Box:
        """
        :param action: dummy action. Next frame is returned.
        :return: observation (raster image) of next frame
        """
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

        return obs, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass
