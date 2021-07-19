import os
import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Tuple

import gym
import numpy as np
import torch
from gym import spaces

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.environment.cle_utils import get_cle, SimulationOutputGym
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
    """
    Custom Environment of L5 Kit that can be registered in OpenAI Gym.
    If closed loop training:
        The raster is updated according to predicted ego positions.
    else (open loop training):
        The raster is based on the ground truth ego positions (i.e. no update).
    """

    def __init__(self) -> None:
        super(L5Env, self).__init__()

        # dataset config
        os.environ["L5KIT_DATA_FOLDER"] = "/home/ubuntu/level5_data/"
        dm = LocalDataManager(None)
        cfg = load_config_data("/home/ubuntu/src/l5kit/l5kit/l5kit/environment/envs/config.yaml")

        # rasterisation
        rasterizer = build_rasterizer(cfg, dm)
        raster_size = cfg["raster_params"]["raster_size"][0]
        n_channels = rasterizer.num_channels()
        print("Using Raster Size: ", raster_size)

        # init dataset
        train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
        self.dataset = EgoDataset(cfg, train_zarr, rasterizer)

        self.future_num_frames = cfg["model_params"]["future_num_frames"]
        self.num_simulation_steps = cfg["gym_params"]["num_simulation_steps"]

        # Define action and observation space
        # Continuous Action Space: gym.spaces.Box (X, Y, Yaw * number of future states)
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(self.future_num_frames * 3, ))

        # Observation Space: gym.spaces.Dict (image: [n_channels, raster_size, raster_size])
        self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=1,
                                              shape=(n_channels, raster_size, raster_size), dtype=np.float32)})

        # Define Close-Loop Simulator
        self.sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                                        distance_th_far=30, distance_th_close=15,
                                        num_simulation_steps=self.num_simulation_steps,
                                        start_frame_index=0, show_info=True)

        self.device = torch.device("cpu")
        self.simulator = ClosedLoopSimulator(self.sim_cfg, self.dataset, self.device)
        self.cle_evaluator = get_cle()

        # Flag for closed-loop training
        self.clt = cfg["gym_params"]["use_cle"]

        # Yaw error (default: True)
        self.use_yaw = cfg["gym_params"]["use_yaw"]
        if self.use_yaw:
            print("Using Yaw")

        # Clip Reward (default: True)
        self.clip_rew = cfg["gym_params"]["clip_rew"]
        self.clip_thresh = 15
        if self.clip_rew:
            print("Clipping Reward")

        # Stop Criterion (default: False)
        self.stop_criterion = cfg["gym_params"]["stop_criterion"]
        self.stop_thresh = 20
        self.stop_error = 0
        if self.stop_criterion:
            print("Episode Stop Criterion Used")

    def set_clt(self, clt: bool) -> None:
        self.clt = clt

    def set_rew_clip(self, rew_clip: float) -> None:
        self.clip_thresh = rew_clip
        print("Clipping Reward at ", self.clip_thresh)

    def reset(self, max_scene_id: int = 99) -> Dict[str, np.ndarray]:
        """
        :param max_scene_id: the maximum scene index to sample from dataset
        :return: the observation of first frame of sampled scene index
        """

        # Sample a episode randomly
        self.scene_indices = [random.randint(0, max_scene_id)]
        self.scene_indices = [0]  # For Overfitting
        self.sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, self.scene_indices, self.sim_cfg)

        # Define in / outs for given scene
        self.agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        self.ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        # Reset CLE evaluator
        self.cle_evaluator.reset()

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
        episode_over = next_frame_index == (len(self.sim_dataset) - self.future_num_frames)  # Note !!

        # EGO
        if not self.sim_cfg.use_ego_gt:
            action = convert_to_dict(action, self.future_num_frames)
            self.ego_output_dict = action

            if self.clt:
                # Update raster according to predicted ego ONLY IF clt
                self.simulator.update_ego(self.sim_dataset, next_frame_index, self.ego_input_dict, self.ego_output_dict)

            ego_frame_in_out = self.simulator.get_ego_in_out(self.ego_input_dict, self.ego_output_dict,
                                                             self.simulator.keys_to_exclude)
            for scene_idx in self.scene_indices:
                self.ego_ins_outs[scene_idx].append(ego_frame_in_out[scene_idx])

        # reward calculation
        reward = self.get_reward()

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

    def get_reward(self) -> float:
        """ Get the reward for the given step.
        The reward is the combination of L2 error from groundtruth trajectory and yaw error
        :return: the reward
        """

        if self.clt:
            assert len(self.scene_indices) == 1

            # generate simulated_outputs
            simulated_outputs: List[SimulationOutputGym] = []
            for scene_idx in self.scene_indices:
                simulated_outputs.append(SimulationOutputGym(scene_idx, self.sim_dataset,
                                                             self.ego_ins_outs, self.agents_ins_outs))
            self.cle_evaluator.evaluate(simulated_outputs)

            # get CLE metrics
            scene_metrics = self.cle_evaluator.scene_metric_results[scene_idx]
            dist_error = scene_metrics['displacement_error_l2'][self.frame_index + 1]
            yaw_error = 3 * torch.abs(scene_metrics['yaw_error_ca'][self.frame_index + 1])

            # clip reward
            if self.clip_rew:
                reward = max(-self.clip_thresh, -dist_error.item())
            else:
                reward = -dist_error.item()

            # use yaw
            if self.use_yaw:
                reward -= yaw_error.item()

            # for early stopping of episode
            self.stop_error = dist_error.item()
        else:
            # Reward for open loop training (MSE)
            penalty = np.square(self.ego_output_dict["positions"] - self.ego_input_dict["target_positions"]).mean()
            reward = - penalty
        return reward

    def check_done_status(self) -> bool:
        """
        (Optionally) End episode if the displacement error crosses a threshold
        :return: end episode flag
        """
        if self.stop_criterion:  # End episode when self.stop_thresh is crossed
            return self.stop_error > self.stop_thresh
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
        pass

    def close(self) -> None:
        pass
