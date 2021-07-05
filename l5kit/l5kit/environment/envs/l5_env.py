import os
import random
from collections import defaultdict
from typing import List, Dict, Optional

import gym
import numpy as np
from gym import spaces
import torch

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import Rasterizer, build_rasterizer
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import ClosedLoopSimulator, UnrollInputOutput, SimulationOutput
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator
from l5kit.environment.cle_utils import SimulationOutputGym, get_cle
from l5kit.environment.utils import default_collate_numpy

class L5Env(gym.Env):
    """
    Custom Gym Environment of L5 Kit.
    This is a full implemented env where a random scene is rolled out. 
    Each frame sequentially is returned as observation.
    An action is taken (dummy/predicted). 
    The raster is updated according to predicted ego positions
    """

    def __init__(self) -> None:
        super(L5Env, self).__init__()
        print("Registering L5Kit Custom Gym Environment")

        # set env variable for data
        os.environ["L5KIT_DATA_FOLDER"] = "/home/ubuntu/level5_data/"

        dm = LocalDataManager(None)
        # get config
        cfg = load_config_data("./envs/config.yaml")

        # rasterisation
        rasterizer = build_rasterizer(cfg, dm)
        raster_size = cfg["raster_params"]["raster_size"][0]
        n_channels = rasterizer.num_channels()
        print("Raster Size: ", raster_size)

        # init dataset
        train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
        self.dataset = EgoDataset(cfg, train_zarr, rasterizer)

        # load pretrained model
        ego_model_path = "/home/ubuntu/models/planning_model_20210421_5steps.pt"
        future_num_frames = torch.load(ego_model_path).model.fc.out_features // 3 # X, Y, Yaw
        # num_simulation_steps = cfg["gym_params"]["num_simulation_steps"]
        num_simulation_steps = None  # !!

        # Define action and observation space
        # Continuous Action Space: gym.spaces.Dict (X, Y, Yaw * number of future states)
        self.action_space = spaces.Dict({
                                'positions': spaces.Box(low=-1000, high=1000, shape=(1, future_num_frames, 2)),
                                'yaws': spaces.Box(low=-1000, high=1000, shape=(1, future_num_frames, 1)),
                                'velocities': spaces.Box(low=-1000, high=1000, shape=(1, future_num_frames, 2))})

        # Observation Space: gym.spaces.Dict (image: [n_channels, raster_size, raster_size])
        self.observation_space = spaces.Dict({
                                    'image': spaces.Box(low=0, high=1,
                                    shape=(n_channels, raster_size, raster_size), dtype=np.float32)})

        # Define Close-Loop Simulator
        # num_simulation_steps = None
        self.sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                                        distance_th_far=30, distance_th_close=15,
                                        num_simulation_steps=num_simulation_steps,
                                        start_frame_index=0, show_info=True)
        
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.simulator = ClosedLoopSimulator(self.sim_cfg, self.dataset, self.device)
        self.cle_evaluator = get_cle()

        # deterministic rollout
        self.det_roll = cfg["gym_params"]["deterministic_rollout"]
        self.det_scene_idx = 0

        # IPC ablation
        raster_out_size = cfg["gym_params"]["raster_out_size"]
        self.raster_out_size = raster_out_size if raster_out_size is not None else raster_size
        print("Raster Out Size: ", self.raster_out_size)
        self.observation_space = spaces.Dict({
                                    'image': spaces.Box(low=0, high=1,
                                    shape=(n_channels, self.raster_out_size, self.raster_out_size), dtype=np.float32)})

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
        obs = {'image': ego_input[0]["image"][:, :self.raster_out_size, :self.raster_out_size]}
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
            obs = {"image": ego_input[0]["image"][:, :self.raster_out_size, :self.raster_out_size]}
            # print("FI: ", self.frame_index, "SI: ", self.scene_indices, "Update: ", should_update)
        else:
            # Dummy final obs (when done is True)
            ego_input = self.sim_dataset.rasterise_frame_batch(0)
            obs = {"image": ego_input[0]["image"][:, :self.raster_out_size, :self.raster_out_size]}

        # done is True when episode ends
        done = not should_update
        if done:
            # generate simulated_outputs
            simulated_outputs: List[SimulationOutputGym] = []
            for scene_idx in self.scene_indices:
                simulated_outputs.append(SimulationOutputGym(scene_idx, self.sim_dataset, self.ego_ins_outs, self.agents_ins_outs))
            self.cle_evaluator.evaluate(simulated_outputs)

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