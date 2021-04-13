import unittest
from typing import Dict
from unittest import mock

import torch

from l5kit.cle import composite_metrics as cm
from l5kit.cle import metrics, validators


class TestPassedDrivenMilesCompositeMetric(unittest.TestCase):
    def test_failed_frames(self) -> None:
        validation_results: Dict[str, validators.ValidatorOutput] = {
            "validator_a": validators.ValidatorOutput(is_valid_scene=False,
                                                      failed_frames=[15])
        }
        metric_results: Dict[str, torch.Tensor] = {
            metrics.SimulatedDrivenMilesMetric.metric_name: torch.ones(20),
        }
        simulation_output = mock.Mock()
        pdm_metric = cm.PassedDrivenMilesCompositeMetric("passed_driven_miles",
                                                         ["validator_a"])
        result = pdm_metric.compute(metric_results, validation_results, simulation_output)
        simulation_output.assert_not_called()
        self.assertEqual(result, 15.0)

    def test_failed_frames_multiple_interventions(self) -> None:
        validation_results: Dict[str, validators.ValidatorOutput] = {
            "validator_a": validators.ValidatorOutput(is_valid_scene=False,
                                                      failed_frames=[10, 11]),
            "validator_b": validators.ValidatorOutput(is_valid_scene=False,
                                                      failed_frames=[15]),
            "validator_c": validators.ValidatorOutput(is_valid_scene=False,
                                                      failed_frames=[15, 18]),
            "validator_d": validators.ValidatorOutput(is_valid_scene=True,
                                                      failed_frames=[])
        }
        metric_results: Dict[str, torch.Tensor] = {
            metrics.SimulatedDrivenMilesMetric.metric_name: torch.ones(20),
        }
        simulation_output = mock.Mock()
        pdm_metric = cm.PassedDrivenMilesCompositeMetric("passed_driven_miles",
                                                         ["validator_a",
                                                          "validator_b",
                                                          "validator_c",
                                                          "validator_d"])
        result = pdm_metric.compute(metric_results, validation_results, simulation_output)
        simulation_output.assert_not_called()
        self.assertEqual(result, 10.0)

    def test_no_failed_frames(self) -> None:
        validation_results: Dict[str, validators.ValidatorOutput] = {
            "validator_a": validators.ValidatorOutput(is_valid_scene=True,
                                                      failed_frames=[])
        }
        metric_results: Dict[str, torch.Tensor] = {
            metrics.SimulatedDrivenMilesMetric.metric_name: torch.ones(20),
        }
        simulation_output = mock.Mock()
        pdm_metric = cm.PassedDrivenMilesCompositeMetric("passed_driven_miles",
                                                         ["validator_a"])
        result = pdm_metric.compute(metric_results, validation_results, simulation_output)
        simulation_output.assert_not_called()
        self.assertEqual(result, 20.0)

    def test_all_failed_frames(self) -> None:
        timesteps = 20
        validation_results: Dict[str, validators.ValidatorOutput] = {
            "validator_a": validators.ValidatorOutput(is_valid_scene=True,
                                                      failed_frames=list(range(timesteps)))
        }
        metric_results: Dict[str, torch.Tensor] = {
            metrics.SimulatedDrivenMilesMetric.metric_name: torch.ones(timesteps),
        }
        simulation_output = mock.Mock()
        pdm_metric = cm.PassedDrivenMilesCompositeMetric("passed_driven_miles",
                                                         ["validator_a"])
        result = pdm_metric.compute(metric_results, validation_results, simulation_output)
        simulation_output.assert_not_called()
        self.assertEqual(result, 0.0)

    def test_ignore_entire_scene(self) -> None:
        validation_results: Dict[str, validators.ValidatorOutput] = {
            "validator_a": validators.ValidatorOutput(is_valid_scene=False,
                                                      failed_frames=[15])
        }
        metric_results: Dict[str, torch.Tensor] = {
            metrics.SimulatedDrivenMilesMetric.metric_name: torch.ones(20),
        }
        simulation_output = mock.Mock()
        pdm_metric = cm.PassedDrivenMilesCompositeMetric("passed_driven_miles",
                                                         ["validator_a"],
                                                         ignore_entire_scene=True)
        result = pdm_metric.compute(metric_results, validation_results, simulation_output)
        self.assertEqual(result, 0.0)


class TestDrivenMilesCompositeMetric(unittest.TestCase):
    def test_zero_miles(self) -> None:
        metric_results: Dict[str, torch.Tensor] = {
            metrics.SimulatedDrivenMilesMetric.metric_name: torch.zeros(10),
        }
        simulation_output = mock.Mock()
        validation_results = mock.Mock()
        dm_metric = cm.DrivenMilesCompositeMetric("total_driven_miles")
        result = dm_metric.compute(metric_results, validation_results, simulation_output)
        simulation_output.assert_not_called()
        validation_results.assert_not_called()
        self.assertEqual(result, 0.0)

    def test_driven_miles(self) -> None:
        # Just ones
        driven_tensor_ones = torch.ones(10)
        metric_results: Dict[str, torch.Tensor] = {
            metrics.SimulatedDrivenMilesMetric.metric_name: driven_tensor_ones,
        }
        simulation_output = mock.Mock()
        validation_results = mock.Mock()
        dm_metric = cm.DrivenMilesCompositeMetric("total_driven_miles")
        result = dm_metric.compute(metric_results, validation_results, simulation_output)
        simulation_output.assert_not_called()
        validation_results.assert_not_called()
        self.assertEqual(result, driven_tensor_ones.sum())

        # Different driven mile each step
        driven_tensor_monotonic = torch.arange(10)
        metric_results[metrics.SimulatedDrivenMilesMetric.metric_name] = driven_tensor_monotonic
        simulation_output.reset_mock()
        validation_results.reset_mock()
        result = dm_metric.compute(metric_results, validation_results, simulation_output)
        simulation_output.assert_not_called()
        validation_results.assert_not_called()
        self.assertEqual(result, driven_tensor_monotonic.sum())


class TestCompositeMetricReductionAggregator(unittest.TestCase):
    def test_aggregate_scenes(self) -> None:
        agg = cm.CompositeMetricAggregator()

        # Scenario: all zeros, 2 composite metrics, 3 scenes
        scene_cm_mock: Dict[int, Dict[str, float]] = {
            0: {"mock_cm_metric1": 0.0},
            1: {"mock_cm_metric2": 0.0},
            2: {"mock_cm_metric2": 0.0}
        }
        agg_scenes = agg.aggregate_scenes(scene_cm_mock)
        self.assertEqual(len(agg_scenes), 2)
        self.assertEqual(agg_scenes["mock_cm_metric1"].item(), 0.0)
        self.assertEqual(agg_scenes["mock_cm_metric2"].item(), 0.0)

        # Scenario: all ones, 2 composite metrics, 3 scenes
        scene_cm_mock = {
            0: {"mock_cm_metric1": 1.0},
            1: {"mock_cm_metric2": 1.0},
            2: {"mock_cm_metric2": 1.0}
        }
        agg_scenes = agg.aggregate_scenes(scene_cm_mock)
        self.assertEqual(len(agg_scenes), 2)
        self.assertEqual(agg_scenes["mock_cm_metric1"].item(), 1.0)
        self.assertEqual(agg_scenes["mock_cm_metric2"].item(), 2.0)
