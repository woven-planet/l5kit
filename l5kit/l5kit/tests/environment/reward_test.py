import unittest
from unittest import mock

import torch

from l5kit.environment import gym_metric_set, reward


class TestL2DisplacementYawReward(unittest.TestCase):
    def test_default_attributes(self) -> None:
        cle_reward = reward.L2DisplacementYawReward()
        self.assertIsInstance(cle_reward.metric_set, gym_metric_set.L2DisplacementYawMetricSet)

    def test_reward_reset(self) -> None:
        attrs = {
            "scene_id": 0,
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.ones(20, 7),
            "get_scene_id.return_value": 0,
        }
        sim_output = mock.Mock(**attrs)
        cle_reward = reward.L2DisplacementYawReward()
        _ = cle_reward.get_reward(0, [sim_output])

        cle_reward.reset()
        self.assertDictEqual(cle_reward.metric_set.evaluator.scene_metric_results, {})

    def test_same_trajectory(self) -> None:
        attrs = {
            "scene_id": 0,
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.ones(20, 7),
            "get_scene_id.return_value": 0,
        }

        cle_reward = reward.L2DisplacementYawReward()
        for i in range(19):
            sim_output = mock.Mock(**attrs)
            cle_reward.reset()
            result = cle_reward.get_reward(i, [sim_output])
            self.assertEqual(result["total"], 0.)
            self.assertEqual(result["distance"], 0.)
            self.assertEqual(result["yaw"], 0.)

    def test_l2_parallel_trajectory(self) -> None:
        attrs = {
            "scene_id": 0,
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
            "get_scene_id.return_value": 0,
        }
        cle_reward = reward.L2DisplacementYawReward(yaw_weight=0.0)
        for i in range(19):
            sim_output = mock.Mock(**attrs)
            cle_reward.reset()
            result = cle_reward.get_reward(i, [sim_output])
            self.assertAlmostEqual(result["total"], -1.4142, 4)
            self.assertAlmostEqual(result["distance"], -1.4142, 4)
            self.assertEqual(result["yaw"], 0.0)

    def test_l2_reward_clipping(self) -> None:
        attrs = {
            "scene_id": 0,
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.full((20, 7), 15.0),
            "get_scene_id.return_value": 0,
        }
        cle_reward = reward.L2DisplacementYawReward(yaw_weight=0.0, rew_clip_thresh=10.0)
        for i in range(19):
            sim_output = mock.Mock(**attrs)
            cle_reward.reset()
            result = cle_reward.get_reward(i, [sim_output])
            self.assertAlmostEqual(result["total"], -10.0, 4)
            self.assertAlmostEqual(result["distance"], -10.0, 4)
            self.assertEqual(result["yaw"], 0.0)

    def test_cle_trajectory_without_clipping(self) -> None:
        attrs = {
            "scene_id": 0,
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
            "get_scene_id.return_value": 0,
        }
        cle_reward = reward.L2DisplacementYawReward(yaw_weight=1.0)
        for i in range(19):
            sim_output = mock.Mock(**attrs)
            cle_reward.reset()
            result = cle_reward.get_reward(i, [sim_output])
            self.assertAlmostEqual(result["total"], -1.4142 - 1.0, 4)
            self.assertAlmostEqual(result["distance"], -1.4142, 4)
            self.assertAlmostEqual(result["yaw"], - 1.0, 4)

    def test_cle_trajectory_with_clipping(self) -> None:
        attrs = {
            "scene_id": 0,
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.full((20, 7), 15.0),
            "get_scene_id.return_value": 0,
        }
        cle_reward = reward.L2DisplacementYawReward(yaw_weight=1.0, rew_clip_thresh=10.0)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732

        for i in range(19):
            sim_output = mock.Mock(**attrs)
            cle_reward.reset()
            result = cle_reward.get_reward(i, [sim_output])
            # l2 reward = 10 (clip)
            self.assertAlmostEqual(result["total"], -10 - (14 - 4 * torch.pi), 4)
            self.assertAlmostEqual(result["distance"], -10.0, 4)
            self.assertAlmostEqual(result["yaw"], - 14 + 4 * torch.pi, 4)
