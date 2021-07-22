import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, NamedTuple

import gym
import numpy as np
import torch
from gym import spaces

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.environment.cle_metricset import SimulationOutputGym
from l5kit.environment.reward import Reward
from l5kit.environment.utils import convert_to_dict, default_collate_numpy
from l5kit.rasterization import build_rasterizer
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import ClosedLoopSimulator, UnrollInputOutput


class GymStepOutput(NamedTuple):
    """ The output dict of gym env.step

    :param obs: the next observation on performing environment step
    :param reward: the reward of the current step
    :param done: flag to indicate end of episode
    :param info: additional information
    """

    obs: Dict[str, np.ndarray]
    reward: int
    done: bool
    info: Dict[str, Any]


class L5Env(gym.Env):
    """ Custom Environment of L5 Kit that can be registered in OpenAI Gym. """

    def __init__(self, env_config_path: Path, sim_cfg: SimulationConfig, reward: Reward, cle: bool) -> None:
        """
        :param env_config_path: path to the L5Kit environment configuration file
        :param simulation_cfg: configuration of the L5Kit closed loop simulator
        :param reward: calculates the reward for the gym environment
        :param cle: flag to enable close loop environment updates
        """
        super(L5Env, self).__init__()

        # env config
        dm = LocalDataManager(None)
        cfg = load_config_data(env_config_path)

        # rasterisation
        rasterizer = build_rasterizer(cfg, dm)
        raster_size = cfg["raster_params"]["raster_size"][0]
        n_channels = rasterizer.num_channels()

        # init dataset
        train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
        self.dataset = EgoDataset(cfg, train_zarr, rasterizer)

        self.future_num_frames = cfg["model_params"]["future_num_frames"]

        # Define action and observation space
        # Continuous Action Space: gym.spaces.Box (X, Y, Yaw * number of future states)
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(self.future_num_frames * 3, ))

        # Observation Space: gym.spaces.Dict (image: [n_channels, raster_size, raster_size])
        self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=1,
                                              shape=(n_channels, raster_size, raster_size), dtype=np.float32)})

        # Simulator Config within Gym
        self.sim_cfg = sim_cfg
        self.simulator = ClosedLoopSimulator(self.sim_cfg, self.dataset, device=torch.device("cpu"),
                                             verify_model=False)

        # Reward
        self.reward = reward

        # Scene Sampling config
        self.max_scene_id = cfg["gym_params"]["max_scene_id"]
        if cfg["gym_params"]["overfit"]:  # Overfit
            self.max_scene_id = 0

        # Flag for close-loop training
        self.cle = cle

    def reset(self) -> Dict[str, np.ndarray]:
        """
        :param max_scene_id: the maximum scene index to sample from dataset
        :return: the observation of first frame of sampled scene index
        """

        # Sample a episode randomly
        self.scene_indices = [random.randint(0, self.max_scene_id)]
        self.sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, self.scene_indices, self.sim_cfg)

        # Define in / outs for given scene
        self.agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        self.ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        # Reset CLE evaluator
        self.reward.reset()

        # Output first observation
        self.frame_index = 0
        ego_input = self.sim_dataset.rasterise_frame_batch(self.frame_index)
        self.ego_input_dict = default_collate_numpy(ego_input[0])

        # Only output the image attribute
        obs = {'image': ego_input[0]["image"]}
        return obs

    def step(self, action: np.ndarray) -> GymStepOutput:
        """
        Analogous to sim_loop.unroll(scenes_to_unroll) in L5Kit

        :param action: the action to perform on current state/frame
        :return: the namedTuple comprising the (next observation, reward, done, info)
                 based on the current action
        """

        frame_index = self.frame_index
        next_frame_index = frame_index + 1
        episode_over = next_frame_index == (len(self.sim_dataset) - self.future_num_frames)

        # EGO
        if not self.sim_cfg.use_ego_gt:
            action = convert_to_dict(action, self.future_num_frames)
            self.ego_output_dict = action

            if self.cle:
                # In closed loop training, the raster is updated according to predicted ego positions.
                self.simulator.update_ego(self.sim_dataset, next_frame_index, self.ego_input_dict, self.ego_output_dict)

            ego_frame_in_out = self.simulator.get_ego_in_out(self.ego_input_dict, self.ego_output_dict,
                                                             self.simulator.keys_to_exclude)
            for scene_idx in self.scene_indices:
                self.ego_ins_outs[scene_idx].append(ego_frame_in_out[scene_idx])

        # reward calculation
        if self.cle:
            reward = self.reward.get_reward(self.frame_index, self.scene_indices, self.sim_dataset,
                                            self.ego_ins_outs, self.agents_ins_outs)
        else:
            reward = self.reward.get_reward(self.ego_output_dict, self.ego_input_dict)

        # done is True when episode ends or error gets too high (optional)
        done = episode_over or self.check_done_status()

        # Optionally we can pass additional info
        # We are using "info" to output simulated outputs
        info = {}
        if done:
            info = {"info": self.get_simulated_outputs()}

        # Prepare next obs
        if not episode_over:
            self.frame_index += 1
            ego_input = self.sim_dataset.rasterise_frame_batch(self.frame_index)
            self.ego_input_dict = default_collate_numpy(ego_input[0])
            obs = {"image": ego_input[0]["image"]}
        else:
            # Dummy final obs (when episode_over)
            ego_input = self.sim_dataset.rasterise_frame_batch(0)
            self.ego_input_dict = default_collate_numpy(ego_input[0])
            obs = {"image": ego_input[0]["image"]}

        # return obs, reward, done, info
        return GymStepOutput(obs, reward, done, info)

    def check_done_status(self) -> bool:
        """
        (Optionally) End episode if the displacement error crosses a threshold
        :return: end episode flag
        """
        if self.reward.stop_flag:
            return self.reward.stop_error > self.reward.stop_thresh
        return False

    def get_simulated_outputs(self) -> List[SimulationOutputGym]:
        """
        Generate simulated outputs at end of episode
        :return: List of simulated outputs
        """
        assert len(self.scene_indices) == 1
        # generate simulated_outputs
        simulated_outputs: List[SimulationOutputGym] = []
        for scene_idx in self.scene_indices:
            simulated_outputs.append(SimulationOutputGym(scene_idx, self.sim_dataset,
                                                         self.ego_ins_outs, self.agents_ins_outs))
        return simulated_outputs

    def render(self) -> None:
        raise NotImplementedError

