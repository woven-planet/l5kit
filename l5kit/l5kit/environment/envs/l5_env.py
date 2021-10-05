import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional

import gym
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.environment.kinematic_model import KinematicModel, UnicycleModel
from l5kit.environment.reward import L2DisplacementYawReward, Reward
from l5kit.environment.utils import (calculate_non_kinematic_rescale_params, KinematicActionRescaleParams,
                                     NonKinematicActionRescaleParams)
from l5kit.rasterization import build_rasterizer
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import (ClosedLoopSimulator, ClosedLoopSimulatorModes, SimulationOutputCLE,
                                     UnrollInputOutput)


#: Maximum acceleration magnitude for kinematic model
MAX_ACC = 6
#: Maximum steer magnitude for kinematic model
MAX_STEER = math.radians(45)


@dataclass
class SimulationConfigGym(SimulationConfig):
    """Defines the default parameters used for the simulation of ego and agents around it in L5Kit Gym.
    Note: num_simulation_steps should be eps_length + 1
    This is because we (may) require to extract the initial speed of the vehicle for the kinematic model
    The speed at start_frame_index is always 0 (not indicative of the true current speed).
    We therefore simulate the episode from (start_frame_index + 1, start_frame_index + eps_length + 1)

    :param use_ego_gt: whether to use GT annotations for ego instead of model's outputs
    :param use_agents_gt: whether to use GT annotations for agents instead of model's outputs
    :param disable_new_agents: whether to disable agents that are not returned at start_frame_index
    :param distance_th_far: if a tracked agent is closed than this value to ego, it will be controlled
    :param distance_th_close: if a new agent is closer than this value to ego, it will be controlled
    :param start_frame_index: the start index of the simulation
    :param num_simulation_steps: the number of step to simulate
    """
    use_ego_gt: bool = False
    use_agents_gt: bool = True
    disable_new_agents: bool = False
    distance_th_far: float = 30.0
    distance_th_close: float = 15.0
    start_frame_index: int = 0
    num_simulation_steps: int = 33


class EpisodeOutputGym(SimulationOutputCLE):
    """This object holds information regarding the simulation output at the end of an episode
    in the gym-compatible L5Kit environment. The output can be used to
    calculate quantitative metrics and provide qualitative visualization.

    :param scene_id: the scene indices
    :param sim_dataset: the simulation dataset
    :param ego_ins_outs: all inputs and outputs for ego (each frame of each scene has only one)
    :param agents_ins_outs: all inputs and outputs for agents (multiple per frame in a scene)
    """

    def __init__(self, scene_id: int, sim_dataset: SimulationDataset,
                 ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]],
                 agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]]):
        """Constructor method
        """
        super(EpisodeOutputGym, self).__init__(scene_id, sim_dataset, ego_ins_outs, agents_ins_outs)

        # Required for Bokeh Visualizer
        simulated_dataset = sim_dataset.scene_dataset_batch[scene_id]
        self.tls_frames = simulated_dataset.dataset.tl_faces
        self.agents_th = simulated_dataset.cfg["raster_params"]["filter_agents_threshold"]


class GymStepOutput(NamedTuple):
    """The output dict of gym env.step

    :param obs: the next observation on performing environment step
    :param reward: the reward of the current step
    :param done: flag to indicate end of episode
    :param info: additional information
    """
    obs: Dict[str, np.ndarray]
    reward: float
    done: bool
    info: Dict[str, Any]


class L5Env(gym.Env):
    """Custom Environment of L5 Kit that can be registered in OpenAI Gym.

    :param env_config_path: path to the L5Kit environment configuration file
    :param dmg: local data manager object
    :param simulation_cfg: configuration of the L5Kit closed loop simulator
    :param train: flag to determine whether to use train or validation dataset
    :param reward: calculates the reward for the gym environment
    :param cle: flag to enable close loop environment updates
    :param rescale_action: flag to rescale the model action back to the un-normalized action space
    :param use_kinematic: flag to use the kinematic model
    :param kin_model: the kinematic model
    :param return_info: flag to return info when a episode ends
    :param randomize_start: flag to randomize the start frame of episode
    """

    def __init__(self, env_config_path: Optional[str] = None, dmg: Optional[LocalDataManager] = None,
                 sim_cfg: Optional[SimulationConfig] = None, train: bool = True,
                 reward: Optional[Reward] = None, cle: bool = True, rescale_action: bool = True,
                 use_kinematic: bool = False, kin_model: Optional[KinematicModel] = None,
                 reset_scene_id: Optional[int] = None, return_info: bool = False,
                 randomize_start: bool = True) -> None:
        """Constructor method
        """
        super(L5Env, self).__init__()

        # Required to register environment
        if env_config_path is None:
            return

        # env config
        dm = dmg if dmg is not None else LocalDataManager(None)
        cfg = load_config_data(env_config_path)
        self.step_time = cfg["model_params"]["step_time"]

        # rasterisation
        rasterizer = build_rasterizer(cfg, dm)
        raster_size = cfg["raster_params"]["raster_size"][0]
        n_channels = rasterizer.num_channels()

        # load dataset of environment
        self.train = train
        self.overfit = cfg["gym_params"]["overfit"]
        self.randomize_start_frame = randomize_start
        if self.train or self.overfit:
            loader_key = cfg["train_data_loader"]["key"]
        else:
            loader_key = cfg["val_data_loader"]["key"]
        dataset_zarr = ChunkedDataset(dm.require(loader_key)).open()
        self.dataset = EgoDataset(cfg, dataset_zarr, rasterizer)

        # Define action and observation space
        # Continuous Action Space: gym.spaces.Box (X, Y, Yaw * number of future states)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3, ))

        # Observation Space: gym.spaces.Dict (image: [n_channels, raster_size, raster_size])
        obs_shape = (n_channels, raster_size, raster_size)
        self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)})

        # Simulator Config within Gym
        self.sim_cfg = sim_cfg if sim_cfg is not None else SimulationConfigGym()
        self.simulator = ClosedLoopSimulator(self.sim_cfg, self.dataset, device=torch.device("cpu"),
                                             mode=ClosedLoopSimulatorModes.GYM)

        self.reward = reward if reward is not None else L2DisplacementYawReward()

        self.max_scene_id = cfg["gym_params"]["max_scene_id"]
        if not self.train:
            self.max_scene_id = cfg["gym_params"]["max_val_scene_id"]
            self.randomize_start_frame = False
        if self.overfit:
            self.overfit_scene_id = cfg["gym_params"]["overfit_id"]

        self.cle = cle
        self.rescale_action = rescale_action
        self.use_kinematic = use_kinematic

        if self.use_kinematic:
            self.kin_model = kin_model if kin_model is not None else UnicycleModel()
            self.kin_rescale = self._get_kin_rescale_params()
        else:
            self.non_kin_rescale = self._get_non_kin_rescale_params()

        # If not None, reset_scene_id is the scene_id that will be rolled out when reset is called
        self.reset_scene_id = reset_scene_id
        if self.overfit:
            self.reset_scene_id = self.overfit_scene_id

        # flag to decide whether to return any info at end of episode
        # helps to limit the IPC
        self.return_info = return_info

        self.seed()

    def seed(self, seed: int = None) -> List[int]:
        """Generate the random seed.

        :param seed: the seed integer
        :return: the output random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        # TODO : add a torch seed for future
        return [seed]

    def set_reset_id(self, reset_id: int = None) -> None:
        """Set the reset_id to unroll from specific scene_id.
        Useful during deterministic evaluation.

        :param reset_id: the scene_id to unroll
        """
        self.reset_scene_id = reset_id

    def reset(self) -> Dict[str, np.ndarray]:
        """ Resets the environment and outputs first frame of a new scene sample.

        :return: the observation of first frame of sampled scene index
        """
        # Define in / outs for new episode scene
        self.agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        self.ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        # Select Scene ID
        self.scene_index = self.np_random.randint(0, self.max_scene_id)
        if self.reset_scene_id is not None:
            self.scene_index = min(self.reset_scene_id, self.max_scene_id - 1)
            self.reset_scene_id += 1

        # Select Frame ID (within bounds of the scene)
        if self.randomize_start_frame:
            scene_length = len(self.dataset.get_scene_indices(self.scene_index))
            self.eps_length = self.sim_cfg.num_simulation_steps or scene_length
            end_frame = scene_length - self.eps_length
            self.sim_cfg.start_frame_index = self.np_random.randint(0, end_frame + 1)

        # Prepare episode scene
        self.sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, [self.scene_index], self.sim_cfg)

        # Reset CLE evaluator
        self.reward.reset()

        # Output first observation
        self.frame_index = 1  # Frame_index 1 has access to the true ego speed
        ego_input = self.sim_dataset.rasterise_frame_batch(self.frame_index)
        self.ego_input_dict = {k: np.expand_dims(v, axis=0) for k, v in ego_input[0].items()}

        # Reset Kinematic model
        if self.use_kinematic:
            init_kin_state = np.array([0.0, 0.0, 0.0, self.step_time * ego_input[0]['curr_speed']])
            self.kin_model.reset(init_kin_state)

        # Only output the image attribute
        obs = {'image': ego_input[0]["image"]}
        return obs

    def step(self, action: np.ndarray) -> GymStepOutput:
        """Inputs the action, updates the environment state and outputs the next frame.

        :param action: the action to perform on current state/frame
        :return: the namedTuple comprising the (next observation, reward, done, info)
            based on the current action
        """
        frame_index = self.frame_index
        next_frame_index = frame_index + 1
        episode_over = next_frame_index == (len(self.sim_dataset) - 1)

        # EGO
        if not self.sim_cfg.use_ego_gt:
            action = self._rescale_action(action)
            ego_output = self._convert_action_to_ego_output(action)
            self.ego_output_dict = ego_output

            if self.cle:
                # In closed loop training, the raster is updated according to predicted ego positions.
                self.simulator.update_ego(self.sim_dataset, next_frame_index, self.ego_input_dict, self.ego_output_dict)

            ego_frame_in_out = self.simulator.get_ego_in_out(self.ego_input_dict, self.ego_output_dict,
                                                             self.simulator.keys_to_exclude)
            self.ego_ins_outs[self.scene_index].append(ego_frame_in_out[self.scene_index])

        # generate simulated_outputs
        simulated_outputs = SimulationOutputCLE(self.scene_index, self.sim_dataset, self.ego_ins_outs,
                                                self.agents_ins_outs)

        # reward calculation
        reward = self.reward.get_reward(self.frame_index, [simulated_outputs])

        # done is True when episode ends
        done = episode_over

        # Optionally we can pass additional info
        # We are using "info" to output rewards and simulated outputs (during evaluation)
        info: Dict[str, Any]
        info = {'reward_tot': reward["total"], 'reward_dist': reward["distance"], 'reward_yaw': reward["yaw"]}
        if done and self.return_info:
            info = {"sim_outs": self.get_episode_outputs(), "reward_tot": reward["total"],
                    "reward_dist": reward["distance"], "reward_yaw": reward["yaw"]}

        # Get next obs
        self.frame_index += 1
        obs = self._get_obs(self.frame_index, episode_over)

        # return obs, reward, done, info
        return GymStepOutput(obs, reward["total"], done, info)

    def get_episode_outputs(self) -> List[EpisodeOutputGym]:
        """Generate and return the outputs at the end of the episode.

        :return: List of episode outputs
        """
        episode_outputs = [EpisodeOutputGym(self.scene_index, self.sim_dataset, self.ego_ins_outs,
                                            self.agents_ins_outs)]
        return episode_outputs

    def render(self) -> None:
        """Render a frame during the simulation
        """
        raise NotImplementedError

    def _get_obs(self, frame_index: int, episode_over: bool) -> Dict[str, np.ndarray]:
        """Get the observation corresponding to a given frame index in the scene.

        :param frame_index: the index of the frame which provides the observation
        :param episode_over: flag to determine if the episode is over
        :return: the observation corresponding to the frame index
        """
        if episode_over:
            frame_index = 0  # Dummy final obs (when episode_over)

        ego_input = self.sim_dataset.rasterise_frame_batch(frame_index)
        self.ego_input_dict = {k: np.expand_dims(v, axis=0) for k, v in ego_input[0].items()}
        obs = {"image": ego_input[0]["image"]}
        return obs

    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        """Rescale the input action back to the un-normalized action space. PPO and related algorithms work well
        with normalized action spaces. The environment receives a normalized action and we un-normalize it back to
        the original action space for environment updates.

        :param action: the normalized action
        :return: the unnormalized action
        """
        if self.rescale_action:
            if self.use_kinematic:
                action[0] = self.kin_rescale.steer_scale * action[0]
                action[1] = self.kin_rescale.acc_scale * action[1]
            else:
                action[0] = self.non_kin_rescale.x_mu + self.non_kin_rescale.x_scale * action[0]
                action[1] = self.non_kin_rescale.y_mu + self.non_kin_rescale.y_scale * action[1]
                action[2] = self.non_kin_rescale.yaw_mu + self.non_kin_rescale.yaw_scale * action[2]
        return action

    def _get_kin_rescale_params(self) -> KinematicActionRescaleParams:
        """Determine the action un-normalization parameters for the kinematic model
        from the current dataset in the L5Kit environment.

        :return: Tuple of the action un-normalization parameters for kinematic model
        """
        global MAX_ACC, MAX_STEER
        return KinematicActionRescaleParams(MAX_STEER * self.step_time, MAX_ACC * self.step_time)

    def _get_non_kin_rescale_params(self, max_num_scenes: int = 10) -> NonKinematicActionRescaleParams:
        """Determine the action un-normalization parameters for the non-kinematic model
        from the current dataset in the L5Kit environment.

        :param max_num_scenes: maximum number of scenes to consider to determine parameters
        :return: Tuple of the action un-normalization parameters for non-kinematic model
        """
        scene_ids = list(range(self.max_scene_id)) if not self.overfit else [self.overfit_scene_id]
        if len(scene_ids) > max_num_scenes:  # If too many scenes, CPU crashes
            scene_ids = scene_ids[:max_num_scenes]
        sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_ids, self.sim_cfg)
        return calculate_non_kinematic_rescale_params(sim_dataset)

    def _convert_action_to_ego_output(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert the input action into ego output format.

        :param action: the input action provided by policy
        :return: action in ego output format, a numpy dict with keys 'positions' and 'yaws'
        """
        if self.use_kinematic:
            data_dict = self.kin_model.update(action[:2])
        else:
            # [batch_size=1, num_steps, (X, Y, yaw)]
            data = action.reshape(1, 1, 3)
            pred_positions = data[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = data[:, :, 2:3]
            data_dict = {"positions": pred_positions, "yaws": pred_yaws}
        return data_dict
