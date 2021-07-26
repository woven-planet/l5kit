import unittest
from typing import Any
from unittest import mock

import torch

from l5kit.cle import metrics
from l5kit.evaluation import error_functions
from l5kit.evaluation import metrics as l5metrics


class TestCollisionMetric(unittest.TestCase):
    @staticmethod
    def create_dummy_metric(dummy_metric_name: str = "dummy_metric") -> Any:
        class DummyMetric(metrics.CollisionMetricBase):
            metric_name = dummy_metric_name

            def __init__(self) -> None:
                super().__init__(l5metrics.CollisionType.FRONT)

        return DummyMetric()

    def test_attributes(self) -> None:
        dummy_metric_name = "dummy_metric"
        dummy_metric = TestCollisionMetric.create_dummy_metric(dummy_metric_name)
        self.assertEqual(dummy_metric.collision_type,
                         l5metrics.CollisionType.FRONT)
        self.assertEqual(dummy_metric.metric_name,
                         dummy_metric_name)

    def test_collision_types(self) -> None:
        collision_metric_match = {
            l5metrics.CollisionType.FRONT: metrics.CollisionFrontMetric(),
            l5metrics.CollisionType.SIDE: metrics.CollisionSideMetric(),
            l5metrics.CollisionType.REAR: metrics.CollisionRearMetric(),
        }
        for collision_type, metric in collision_metric_match.items():
            self.assertEqual(metric.collision_type, collision_type)


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


class TestDistanceToRefTrajectory(unittest.TestCase):
    def test_same_trajectory(self) -> None:
        attrs = {
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.ones(20, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DistanceToRefTrajectoryMetric()
        result = metric.compute(sim_output)
        self.assertEqual(result.sum(), 0.)

    def test_different_fraction(self) -> None:
        simulated_steps = 20
        scene_fraction = 0.5
        attrs = {
            "simulated_ego_states": torch.ones(simulated_steps, 7),
            "recorded_ego_states": torch.ones(simulated_steps, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DistanceToRefTrajectoryMetric(scene_fraction)
        result = metric.compute(sim_output)

        simulated_steps_fraction = int(simulated_steps * scene_fraction)
        self.assertEqual(len(result), simulated_steps_fraction)
        self.assertEqual(result.sum(), 0.)

    def test_symmetry(self) -> None:
        attrs = {
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DistanceToRefTrajectoryMetric()
        result = metric.compute(sim_output)

        attrs = {
            "simulated_ego_states": torch.full((20, 7), 2.0),
            "recorded_ego_states": torch.ones(20, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DistanceToRefTrajectoryMetric()
        result_switch = metric.compute(sim_output)
        self.assertEqual(result.sum(), result_switch.sum())

    def test_parallel_trajectory(self) -> None:
        simulated_steps = 20
        attrs = {
            "simulated_ego_states": torch.ones(simulated_steps, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DistanceToRefTrajectoryMetric()
        result = metric.compute(sim_output)

        # Default fraction should be 80% of the samples
        simulated_steps_fraction = int(simulated_steps * 0.8)
        self.assertEqual(len(result), simulated_steps_fraction)
        self.assertEqual(len(torch.unique(result)), 1)
        self.assertAlmostEqual(torch.unique(result).item(), 1.4142, 4)

    def test_larger_observed_ego(self) -> None:
        simulated_steps = 20
        attrs = {
            "simulated_ego_states": torch.ones(simulated_steps, 7),
            "recorded_ego_states": torch.ones(50, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DistanceToRefTrajectoryMetric()
        result = metric.compute(sim_output)
        self.assertEqual(result.sum(), 0.)

        # Default fraction should be 80% of the samples
        simulated_steps_fraction = int(simulated_steps * 0.8)
        self.assertEqual(len(result), simulated_steps_fraction)

    def test_larger_simulated_ego(self) -> None:
        attrs = {
            "simulated_ego_states": torch.ones(50, 7),
            "recorded_ego_states": torch.ones(20, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DistanceToRefTrajectoryMetric()

        with self.assertRaisesRegex(ValueError, "More simulated timesteps than observed"):
            _ = metric.compute(sim_output)

    def test_half_trajectories(self) -> None:
        observed_trajectory = torch.ones(40, 7)
        observed_trajectory[20:, :] += 1.0
        attrs = {
            "simulated_ego_states": torch.ones(40, 7),
            "recorded_ego_states": observed_trajectory,
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DistanceToRefTrajectoryMetric()
        result = metric.compute(sim_output)
        self.assertEqual(len(torch.unique(result)), 1)

    def test_more_simulation_than_observation(self) -> None:
        timesteps = 20
        attrs = {
            "simulated_ego_states": torch.ones(timesteps + 20, 7),
            "recorded_ego_states": torch.ones(timesteps, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.DistanceToRefTrajectoryMetric()
        with self.assertRaisesRegex(ValueError, "More simulated timesteps than observed"):
            _ = metric.compute(sim_output)


class TestSimulatedDrivenMilesMetric(unittest.TestCase):
    def test_no_movement_trajectory(self) -> None:
        timesteps = 20
        attrs = {
            "simulated_ego_states": torch.ones(timesteps, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.SimulatedDrivenMilesMetric()
        result = metric.compute(sim_output)
        self.assertEqual(result.size(0), timesteps)
        self.assertEqual(result.sum().item(), 0.0)

    def test_one_axis_movement_trajectory(self) -> None:
        timesteps = 20
        attrs = {
            "simulated_ego_states": torch.ones(timesteps, 7),
        }
        # Set one coordinate to always 1 and keep the other
        # increasing
        increasing_tensor = torch.tensor([i for i in range(timesteps)])
        attrs["simulated_ego_states"][..., 1] += increasing_tensor
        sim_output = mock.Mock(**attrs)
        metric = metrics.SimulatedDrivenMilesMetric()
        result = metric.compute(sim_output)
        self.assertEqual(result.size(0), timesteps)

        # How much is moved for each frame in miles (one meter per frame)
        single_step_miles = 1.0 * metrics.SimulatedDrivenMilesMetric.METER_TO_MILES
        expected_driven_miles = single_step_miles * (timesteps - 1)
        self.assertAlmostEqual(result.sum().item(),
                               expected_driven_miles, places=3)
        # Should have only a zero for the first step and then
        # the same step for other frames
        self.assertEqual(len(result.unique()), 2)


class TestReplayDrivenMilesMetric(unittest.TestCase):
    def test_no_movement_trajectory(self) -> None:
        timesteps = 20
        attrs = {
            "recorded_ego_states": torch.ones(timesteps, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.ReplayDrivenMilesMetric()
        result = metric.compute(sim_output)
        self.assertEqual(result.size(0), timesteps)
        self.assertEqual(result.sum().item(), 0.0)

    def test_one_axis_movement_trajectory(self) -> None:
        timesteps = 20
        tensor_ego_states = torch.ones(timesteps, 7)
        # Set one coordinate to always 1 and keep the other
        # increasing
        tensor_ego_states[:, 0] += torch.arange(0, timesteps)
        attrs = {
            "recorded_ego_states": tensor_ego_states,
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.ReplayDrivenMilesMetric()
        result = metric.compute(sim_output)
        self.assertEqual(result.size(0), timesteps)

        # How much is moved for each frame in miles (one meter per frame)
        single_step_miles = 1.0 * metrics.ReplayDrivenMilesMetric.METER_TO_MILES
        expected_driven_miles = single_step_miles * (timesteps - 1)
        self.assertAlmostEqual(result.sum().item(),
                               expected_driven_miles, places=3)
        # Should have only a zero for the first step and then
        # the same step for other frames
        self.assertEqual(len(result.unique()), 2)


class TestYawErrorMetric(unittest.TestCase):
    def test_same_trajectory(self) -> None:
        timesteps = 20
        attrs = {
            "simulated_ego_states": torch.ones(timesteps, 7),
            "recorded_ego_states": torch.ones(timesteps, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.YawErrorMetric(error_functions.closest_angle_error)
        result = metric.compute(sim_output)
        self.assertEqual(len(result), timesteps)
        self.assertEqual(result.sum(), 0.)

    def test_rotating_trajectory(self) -> None:
        attrs = {
            "simulated_ego_states": torch.zeros(20, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.YawErrorMetric(error_functions.closest_angle_error)
        result = metric.compute(sim_output)
        self.assertEqual(len(torch.unique(result)), 1)
        self.assertAlmostEqual(torch.unique(result).item(), 2.0000, 4)

    def test_wrap_rotating_trajectory(self) -> None:
        attrs = {
            "simulated_ego_states": torch.zeros(20, 7),
            "recorded_ego_states": torch.full((20, 7), 4.0),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.YawErrorMetric(error_functions.closest_angle_error)
        result = metric.compute(sim_output)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
        self.assertEqual(len(torch.unique(result)), 1)
        self.assertAlmostEqual(torch.unique(result).item(), 2 * torch.pi - 4, 4)

    def test_yaw_closest_angle_rotating_trajectory(self) -> None:
        attrs = {
            "simulated_ego_states": torch.zeros(20, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
        }
        sim_output = mock.Mock(**attrs)
        metric_ca_arg = metrics.YawErrorMetric(error_functions.closest_angle_error)
        result_ca_arg = metric_ca_arg.compute(sim_output)

        metric = metrics.YawErrorCAMetric()
        result = metric.compute(sim_output)

        # Make sure both results match
        self.assertTrue((result_ca_arg == result).all())

    def test_half_trajectories(self) -> None:
        observed_trajectory = torch.ones(40, 7)
        observed_trajectory[20:, :] += 1.0
        attrs = {
            "simulated_ego_states": torch.ones(40, 7),
            "recorded_ego_states": observed_trajectory,
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.YawErrorMetric(error_functions.closest_angle_error)
        result = metric.compute(sim_output)
        self.assertEqual(len(torch.unique(result)), 2)

    def test_symmetry(self) -> None:
        attrs = {
            "simulated_ego_states": torch.ones(20, 7),
            "recorded_ego_states": torch.full((20, 7), 2.0),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.YawErrorMetric(error_functions.closest_angle_error)
        result = metric.compute(sim_output)

        attrs = {
            "simulated_ego_states": torch.full((20, 7,), 2.0),
            "recorded_ego_states": torch.ones(20, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.YawErrorMetric(error_functions.closest_angle_error)
        result_switch = metric.compute(sim_output)
        self.assertEqual(result.sum(), result_switch.sum())

    def test_more_simulation_than_observation(self) -> None:
        timesteps = 20
        attrs = {
            "simulated_ego_states": torch.ones(timesteps + 20, 7),
            "recorded_ego_states": torch.ones(timesteps, 7),
        }
        sim_output = mock.Mock(**attrs)
        metric = metrics.YawErrorMetric(error_functions.closest_angle_error)
        with self.assertRaisesRegex(ValueError, "More simulated timesteps than observed"):
            _ = metric.compute(sim_output)
