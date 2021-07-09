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
from l5kit.cle.metrics import DisplacementErrorL2Metric

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
        cfg = load_config_data("/home/ubuntu/src/l5kit/l5kit/l5kit/environment/envs/config.yaml")

        # rasterisation
        rasterizer = build_rasterizer(cfg, dm)
        raster_size = cfg["raster_params"]["raster_size"][0]
        n_channels = rasterizer.num_channels()
        print("Raster Size: ", raster_size)

        # init dataset
        train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
        self.dataset = EgoDataset(cfg, train_zarr, rasterizer)

        # load pretrained model
        self.future_num_frames = cfg["model_params"]["future_num_frames"]
        if cfg["model_params"]["load_pretrained"]:
            print("Loading pretrained model.....")
            ego_model_path = "/home/ubuntu/models/planning_model_20210421_5steps.pt"
            self.future_num_frames = torch.load(ego_model_path).model.fc.out_features // 3   # X, Y, Yaw

        # num_simulation_steps = cfg["gym_params"]["num_simulation_steps"]
        # num_simulation_steps = None  # !!
        num_simulation_steps = 16  # Stop after Frame 16!!

        # Define action and observation space
        # Continuous Action Space: gym.spaces.Box (X, Y, Yaw * number of future states)
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(self.future_num_frames * 3, ))

        # Observation Space: gym.spaces.Dict (image: [n_channels, raster_size, raster_size])
        self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=1,
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
        self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=1,
                                              shape=(n_channels, self.raster_out_size, self.raster_out_size),
                                              dtype=np.float32)})
        self.clt = False

    def reset(self, max_scene_id: int = 99) -> Dict[str, np.ndarray]:
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
            self.scene_indices = [0]
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
        obs = {'image': ego_input[0]["image"][:, :self.raster_out_size, :self.raster_out_size]}
        # print("FI: ", self.frame_index, "SI: ", self.scene_indices)
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
        should_update = next_frame_index != len(self.sim_dataset)

        # EGO
        if not self.sim_cfg.use_ego_gt:
            action = convert_to_dict(action, self.future_num_frames)
            self.ego_output_dict = action

            if should_update and self.clt:
                self.simulator.update_ego(self.sim_dataset, next_frame_index, self.ego_input_dict, self.ego_output_dict)

            ego_frame_in_out = self.simulator.get_ego_in_out(self.ego_input_dict, self.ego_output_dict,
                                                             self.simulator.keys_to_exclude)
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

        # reward calculation
        reward, self.dist2ref_error = self.get_reward()

        # done is True when episode ends or error gets too high
        done = not should_update or self.check_done_status()

        # if not done:
        #     reward = 0
        # print("Reward: ", done, reward, self.dist2ref_error)

        # Optionally we can pass additional info, we are using that to output simulated outputs
        info = {}
        if done:
            info = {"info": self.get_info()}

        # return obs, reward, done, info
        return GymStepOutput(obs, reward, done, info)

    def get_reward(self) -> Tuple[float, float]:
        dist2ref_error = 0.0
        if self.clt:
            assert len(self.scene_indices) == 1
            # generate simulated_outputs
            simulated_outputs: List[SimulationOutputGym] = []
            for scene_idx in self.scene_indices:
                simulated_outputs.append(SimulationOutputGym(scene_idx, self.sim_dataset,
                                                            self.ego_ins_outs, self.agents_ins_outs))
            self.cle_evaluator.evaluate(simulated_outputs)
            dist_error = self.cle_evaluator.scene_metric_results[scene_idx]['displacement_error_l2'][self.frame_index]
            dist2ref_error = self.cle_evaluator.scene_metric_results[scene_idx]['distance_to_reference_trajectory'][self.frame_index]
            # print(self.cle_evaluator.scene_metric_results[scene_idx]['displacement_error_l2'][:15])
            reward = - dist_error.item()
            dist2ref_error = dist2ref_error.item()
        else:
            penalty = np.square(self.ego_output_dict["positions"] - self.ego_input_dict["target_positions"]).mean()
            reward = - penalty
        return reward, dist2ref_error

    def check_done_status(self, mode: Optional[str] = None, dist2ref_thresh: Optional[float] = 4.0) -> bool:
        if mode is None:  # do nothing, continue
            return False

        if mode == 'dist2ref':  # End episode when dist2ref_thresh is crossed
            return self.dist2ref_error > dist2ref_thresh

    def get_info(self) -> List[SimulationOutputGym]:
        assert len(self.scene_indices) == 1
        # generate simulated_outputs
        simulated_outputs: List[SimulationOutputGym] = []
        for scene_idx in self.scene_indices:
            simulated_outputs.append(SimulationOutputGym(scene_idx, self.sim_dataset,
                                                         self.ego_ins_outs, self.agents_ins_outs))
        return simulated_outputs

    def render(self, mode: Optional[str] = 'console') -> None:
        pass

    def close(self) -> None:
        pass
