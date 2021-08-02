import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional

import gym
import numpy as np
import torch
from gym import spaces

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.environment.kinematic_model import KinematicModel, UnicycleModel
from l5kit.environment.reward import CLEReward, Reward
from l5kit.environment.utils import ActionRescaleParams, calculate_rescale_params, convert_to_numpy
from l5kit.rasterization import build_rasterizer
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import ClosedLoopSimulator, SimulationOutputCLE, UnrollInputOutput


class SimulationConfigGym(SimulationConfig):
    """Defines the default parameters used for the simulation of ego and agents around it in L5Kit Gym.

    :param eps_length: the number of step to simulate per episode in the gym environment.
    """

    def __new__(cls, eps_length: int = 32, start_frame_idx: int = 0) -> 'SimulationConfigGym':
        """Constructor method
        """
        # Note: num_simulation_steps = eps_length + 1
        # This is because we (may) require to extract the initial speed of the vehicle for the kinematic model
        # The speed at start_frame_idx is always 0 (not indicative of the true current speed).
        # We therefore simulate the episode from (start_frame_idx + 1, start_frame_idx + eps_length + 1)
        self = super(SimulationConfigGym, cls).__new__(cls, use_ego_gt=False, use_agents_gt=True,
                                                       disable_new_agents=False, distance_th_far=30,
                                                       distance_th_close=15, start_frame_index=start_frame_idx,
                                                       num_simulation_steps=eps_length + 1)
        return self


class SimulationOutputGym(SimulationOutputCLE):
    """This object holds information about the result of the simulation loop
    for a given scene dataset in gym-compatible L5Kit environment.

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
        super(SimulationOutputGym, self).__init__(scene_id, sim_dataset, ego_ins_outs, agents_ins_outs)

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
    :param reward: calculates the reward for the gym environment
    :param cle: flag to enable close loop environment updates
    :param rescale_action: flag to rescale the model action back to the un-normalized action space
    :param use_kinematic: flag to use the kinematic model
    :param kin_model: the kinematic model
    """

    def __init__(self, env_config_path: str, dmg: Optional[LocalDataManager] = None,
                 sim_cfg: Optional[SimulationConfig] = None,
                 reward: Optional[Reward] = None, cle: bool = True, rescale_action: bool = True,
                 use_kinematic: bool = False, kin_model: Optional[KinematicModel] = None,
                 reset_scene_id: Optional[int] = None) -> None:
        """Constructor method
        """
        super(L5Env, self).__init__()

        # env config
        dm = dmg if dmg is not None else LocalDataManager(None)
        cfg = load_config_data(env_config_path)

        # rasterisation
        rasterizer = build_rasterizer(cfg, dm)
        raster_size = cfg["raster_params"]["raster_size"][0]
        n_channels = rasterizer.num_channels()

        # init dataset
        train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
        self.dataset = EgoDataset(cfg, train_zarr, rasterizer)

        # Define action and observation space
        # Continuous Action Space: gym.spaces.Box (X, Y, Yaw * number of future states)
        # self.action_space = spaces.Box(low=-1000, high=1000, shape=(3, ))
        # self.action_space = spaces.Box(low=-2, high=2, shape=(3, ))
        self.action_space = spaces.Box(low=-1, high=1, shape=(3, ))

        # Observation Space: gym.spaces.Dict (image: [n_channels, raster_size, raster_size])
        obs_shape = (n_channels, raster_size, raster_size)
        self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)})

        # Simulator Config within Gym
        self.sim_cfg = sim_cfg if sim_cfg is not None else SimulationConfigGym()
        self.simulator = ClosedLoopSimulator(self.sim_cfg, self.dataset, device=torch.device("cpu"),
                                             verify_model=False)

        self.reward = reward if reward is not None else CLEReward()

        self.max_scene_id = cfg["gym_params"]["max_scene_id"]
        self.overfit = cfg["gym_params"]["overfit"]
        if self.overfit:
            self.overfit_scene_id = cfg["gym_params"]["overfit_id"]

        self.cle = cle
        self.rescale_action = rescale_action
        self.use_kinematic = use_kinematic

        self.action_rescale_params = self._get_action_rescale_params()

        if self.use_kinematic:
            self.kin_model = kin_model if kin_model is not None else UnicycleModel()

        # If not None, reset_scene_id is the scene_id that will be rolled out when reset is called
        self.reset_scene_id = reset_scene_id
        if self.overfit:
            self.reset_scene_id = self.overfit_scene_id

    def reset(self) -> Dict[str, np.ndarray]:
        """ Resets the environment and outputs first frame of a new scene sample.

        :return: the observation of first frame of sampled scene index
        """
        # Define in / outs for new episode scene
        self.agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        self.ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        if self.reset_scene_id is not None:
            self.scene_index = self.reset_scene_id
        else:  # Sample a scene
            self.scene_index = random.randint(0, self.max_scene_id)
        self.sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, [self.scene_index], self.sim_cfg)

        # Reset CLE evaluator
        self.reward.reset()

        # Output first observation
        self.frame_index = 1  # Frame_index 1 has access to the true ego speed
        ego_input = self.sim_dataset.rasterise_frame_batch(self.frame_index)
        self.ego_input_dict = convert_to_numpy(ego_input[0])

        # Reset Kinematic model
        if self.use_kinematic:
            init_kin_state = np.array([0.0, 0.0, 0.0, 0.1 * ego_input[0]['curr_speed']])
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
        # We are using "info" to output simulated outputs
        info = {}
        if done:
            info = {"info": self.get_simulated_outputs()}

        # Get next obs
        self.frame_index += 1
        obs = self._get_obs(self.frame_index, episode_over)

        # return obs, reward, done, info
        return GymStepOutput(obs, reward, done, info)

    def get_simulated_outputs(self) -> List[SimulationOutputGym]:
        """Generate and output the simulation outputs for the episode.

        :return: List of simulated outputs
        """
        # generate simulated_outputs
        simulated_outputs = [SimulationOutputGym(self.scene_index, self.sim_dataset, self.ego_ins_outs,
                                                 self.agents_ins_outs)]
        return simulated_outputs

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
        self.ego_input_dict = convert_to_numpy(ego_input[0])
        obs = {"image": ego_input[0]["image"]}
        return obs

    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        """Rescale the input action back to the un-normalized action space. PPO and related algorithms work well
        with normalized action spaces. The environment receives a normalized action and we un-normalize it back to
        the original action space for environment updates.

        :param action: the normalized action
        :return: the unnormalized action
        """
        assert len(action) == 3
        if self.rescale_action:
            if self.use_kinematic:
                action[0] = self.action_rescale_params.steer_scale * action[0]
                action[1] = self.action_rescale_params.acc_scale * action[1]
            else:
                action[0] = self.action_rescale_params.x_mu + self.action_rescale_params.x_scale * action[0]
                action[1] = self.action_rescale_params.y_mu + self.action_rescale_params.y_scale * action[1]
                action[2] = self.action_rescale_params.yaw_mu + self.action_rescale_params.yaw_scale * action[2]
        return action

    def _get_action_rescale_params(self) -> ActionRescaleParams:
        """Determine the action un-normalization parameters for the current dataset in the L5Kit environment.

        :return: Tuple of the action un-normalization parameters
        """
        scene_ids = list(range(self.max_scene_id + 1)) if not self.overfit else [self.overfit_scene_id]
        sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_ids, self.sim_cfg)
        return calculate_rescale_params(sim_dataset, self.use_kinematic)

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
