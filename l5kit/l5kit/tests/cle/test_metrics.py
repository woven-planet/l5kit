import unittest
from unittest import mock

import torch

from l5kit.cle import metrics
from l5kit.evaluation import error_functions


class TestDisplacementErrorMetric(unittest.TestCase):
    def test_same_trajectory(self) -> None:
        timesteps = 20
        attrs = {
            "simulated_ego_states": torch.ones(timesteps, 7),
            "recorded_ego_states": torch.ones(timesteps, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DisplacementErrorMetric(error_functions.l2_error)
        result = metric.compute(sim_output)
        self.assertEqual(len(result), timesteps)
        self.assertEqual(result.sum(), 0.)

    def test_parallel_trajectory(self) -> None:
        attrs = {
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DisplacementErrorMetric(error_functions.l2_error)
        result = metric.compute(sim_output)
        self.assertEqual(len(torch.unique(result)), 1)
        self.assertAlmostEqual(torch.unique(result).item(), 1.4142, 4)

    def test_l2_distance_parallel_trajectory(self) -> None:
        attrs = {
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
        }
        sim_output = mock.Mock(**attrs)
        metric_l2_arg = metrics.DisplacementErrorMetric(error_functions.l2_error)
        result_l2_arg = metric_l2_arg.compute(sim_output)

        metric = metrics.DisplacementErrorL2Metric()
        result = metric.compute(sim_output)

        # Make sure both results match
        self.assertTrue((result_l2_arg == result).all())

    def test_half_trajectories(self) -> None:
        observed_trajectory = torch.ones(40, 7)
        observed_trajectory[20:, :] += 1.0
        attrs = {
            "simulated_ego_states": torch.ones(40, 7),
            "recorded_ego_states": observed_trajectory,
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DisplacementErrorMetric(error_functions.l2_error)
        result = metric.compute(sim_output)
        # This is mainly where displacement diverges from distance to ref traj
        self.assertEqual(len(torch.unique(result)), 2)

    def test_symmetry(self) -> None:
        attrs = {
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DisplacementErrorMetric(error_functions.l2_error)
        result = metric.compute(sim_output)

        attrs = {
            "simulated_ego_states": torch.full((20, 7,), 2.0),
            "recorded_ego_states": torch.ones(20, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DisplacementErrorMetric(error_functions.l2_error)
        result_switch = metric.compute(sim_output)
        self.assertEqual(result.sum(), result_switch.sum())

    def test_more_simulation_than_observation(self) -> None:
        timesteps = 20
        attrs = {
            "simulated_ego_states": torch.ones(timesteps + 20, 7),
            "recorded_ego_states": torch.ones(timesteps, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DisplacementErrorMetric(error_functions.l2_error)
        with self.assertRaisesRegex(ValueError, "More simulated timesteps than observed"):
            _ = metric.compute(sim_output)
