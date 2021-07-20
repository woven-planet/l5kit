from typing import DefaultDict, Dict, List, Optional

import numpy as np
import torch

from l5kit.environment.cle_utils import get_cle, SimulationOutputGym
from l5kit.simulation.dataset import SimulationDataset
from l5kit.simulation.unroll import UnrollInputOutput


class Reward(object):
    def __init__(self, rew_clip_thresh: Optional[float] = 15, use_yaw: Optional[bool] = True,
                 yaw_weight: Optional[float] = 3.0, open_loop: Optional[bool] = False,
                 stop_thresh: Optional[float] = 100000) -> None:
        """
        :param rew_clip_thresh: the threshold to clip the reward
        :param use_yaw: flag to penalize the yaw prediction
        :param yaw_weight: weight of the yaw error
        :param open_loop: flag to calculate open loop reward
        :param stop_thresh: the reward threshold to early terminate an episode
        """

        # To add Metric Set
        self.closed_loop_evaluator = get_cle()
        self.open_loop = open_loop
        self.use_yaw = use_yaw
        self.yaw_weight = yaw_weight
        self.rew_clip_thresh = rew_clip_thresh
        self.stop_thresh = stop_thresh

    def reset(self):
        """
        Reset the closed loop evaluator when a new episode starts
        """
        self.closed_loop_evaluator.reset()

    def get_close_loop_reward(self, frame_index: int, scene_indices: List[int], sim_dataset: SimulationDataset,
                              ego_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]],
                              agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]]) -> float:
        """ Get the reward for the given step in close loop training.
        :param frame_index: the current frame index of the episode. Prediction is made for this frame.
        :param scene_indices: the scene indices being rolled out in the environment
        :param sim_dataset: the simulation dataset object of the environment
        :param ego_ins_outs: object contain the ground truth and prediction information of the ego
        :param agents_ins_outs: object contain the ground truth and prediction information of the agents

        :return: the reward is the combination of L2 error from groundtruth trajectory and (optionally) yaw error
        """

        assert len(scene_indices) == 1

        # generate simulated_outputs
        simulated_outputs: List[SimulationOutputGym] = []
        for scene_idx in scene_indices:
            simulated_outputs.append(SimulationOutputGym(scene_idx, sim_dataset, ego_ins_outs, agents_ins_outs))
        self.closed_loop_evaluator.evaluate(simulated_outputs)

        # get CLE metrics
        scene_metrics = self.closed_loop_evaluator.scene_metric_results[scene_idx]
        dist_error = scene_metrics['displacement_error_l2'][frame_index + 1]
        yaw_error = self.yaw_weight * torch.abs(scene_metrics['yaw_error_ca'][frame_index + 1])

        # clip reward
        reward = max(-self.rew_clip_thresh, -dist_error.item())

        # use yaw
        if self.use_yaw:
            reward -= yaw_error.item()

        # for early stopping of episode
        self.stop_error = dist_error.item()

        return reward

    def get_open_loop_reward(self, ego_output_dict: Dict[str, np.ndarray],
                             ego_input_dict: Dict[str, np.ndarray]) -> float:
        """ Get the reward for the given step in open loop training.
        :param ego_output_dict: dictionary containing the predicted ego positions and yaws
        :param ego_input_dict: dictionary containing the target ego positions and yaws
        :return: the reward is the L2 error from groundtruth trajectory
        """
        # Reward for open loop training (MSE)
        penalty = np.square(ego_output_dict["positions"] - ego_input_dict["target_positions"]).mean()
        reward = - penalty

        # for early stopping of episode
        self.stop_error = penalty

        return reward
