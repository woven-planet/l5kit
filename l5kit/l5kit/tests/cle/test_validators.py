import unittest
from typing import Dict
from unittest import mock

import torch

from l5kit.cle import validators


class TestRangeValidator(unittest.TestCase):
    def test_cumsum_with_reset(self) -> None:
        ts_diff = torch.full((20,), 0.1, dtype=torch.float32)
        validation_mask = torch.zeros(20, dtype=torch.bool)
        validation_mask[0:8] = True
        validation_mask[10:] = True

        cumsum = validators.RangeValidator.cumsum_with_reset(ts_diff, validation_mask)
        self.assertEqual((cumsum > 0.5).sum(), 8)

    def test_cumsum_with_reset_no_timestamps(self) -> None:
        ts_diff = torch.zeros(20, dtype=torch.float32)
        validation_mask = torch.zeros(20, dtype=torch.bool)
        cumsum = validators.RangeValidator.cumsum_with_reset(ts_diff, validation_mask)
        self.assertEqual(cumsum.sum(), 0.0)

        validation_mask = torch.ones(20, dtype=torch.bool)
        cumsum = validators.RangeValidator.cumsum_with_reset(ts_diff, validation_mask)
        self.assertEqual(cumsum.sum(), 0.0)

    def test_cumsum_with_reset_with_timestamps(self) -> None:
        ts_diff = torch.ones(20, dtype=torch.float32)

        validation_mask = torch.zeros(20, dtype=torch.bool)
        cumsum = validators.RangeValidator.cumsum_with_reset(ts_diff, validation_mask)
        self.assertEqual(cumsum.sum(), 0.0)

        # Should match pytorch implementation in this case
        validation_mask = torch.ones(20, dtype=torch.bool)
        cumsum = validators.RangeValidator.cumsum_with_reset(ts_diff, validation_mask)
        self.assertEqual(cumsum.sum(), torch.cumsum(ts_diff, dim=0).sum())


class TestValidationCountingAggregator(unittest.TestCase):
    def test_aggregate_scenes(self) -> None:
        agg = validators.ValidationCountingAggregator()
        mock_validator_output = mock.Mock()
        is_valid_scene = mock.PropertyMock(return_value=False)
        type(mock_validator_output).is_valid_scene = is_valid_scene
        scene_validation_mock: Dict[int, Dict[str, validators.ValidatorOutput]] = {
            0: {"mock_validator1": mock_validator_output},
            1: {"mock_validator2": mock_validator_output},
            2: {"mock_validator2": mock_validator_output}
        }
        agg_scenes = agg.aggregate_scenes(scene_validation_mock)
        is_valid_scene.assert_called()

        self.assertEqual(len(agg_scenes), 2)
        self.assertEqual(agg_scenes["mock_validator1"].item(), 1)
        self.assertEqual(agg_scenes["mock_validator2"].item(), 2)

    def test_aggregate_count_failed_frames(self) -> None:
        agg = validators.ValidationCountingAggregator(failed_frames=True)
        mock_validator_output = mock.Mock()
        is_valid_scene = mock.PropertyMock(return_value=False)
        failed_frames = mock.PropertyMock(return_value=[1, 2, 3, 4])
        type(mock_validator_output).is_valid_scene = is_valid_scene
        type(mock_validator_output).failed_frames = failed_frames
        scene_validation_mock: Dict[int, Dict[str, validators.ValidatorOutput]] = {
            0: {"mock_validator1": mock_validator_output},
            1: {"mock_validator2": mock_validator_output},
            2: {"mock_validator2": mock_validator_output}
        }
        agg_scenes = agg.aggregate_scenes(scene_validation_mock)
        failed_frames.assert_called()

        self.assertEqual(len(agg_scenes), 2)
        self.assertEqual(agg_scenes["mock_validator1"].item(), 4)
        self.assertEqual(agg_scenes["mock_validator2"].item(), 8)


class TestValidationFrameAggregator(unittest.TestCase):
    def test_aggregate_scenes(self) -> None:
        agg = validators.ValidationFailedFramesAggregator()

        mock_validator_output = mock.Mock()
        failed_frames = mock.PropertyMock(return_value=[1, 2, 3, 4])
        type(mock_validator_output).failed_frames = failed_frames

        # 3 scenes
        scene_validation_mock: Dict[int, Dict[str, validators.ValidatorOutput]] = {
            0: {"mock_validator1": mock_validator_output},
            1: {"mock_validator2": mock_validator_output},
            2: {"mock_validator2": mock_validator_output}
        }
        agg_scenes = agg.aggregate_scenes(scene_validation_mock)
        failed_frames.assert_called()

        # Size of tensor should be:
        # 4 failed scene frames, given that we have only 1 scene
        print(agg_scenes)
        self.assertEqual(agg_scenes["mock_validator1"].size(0), 4)

        # Size of tensor should be:
        # 4 failed scene frames * 2 (scenes)
        # Given that we have 2 scenes for this validator
        self.assertEqual(agg_scenes["mock_validator2"].size(0), 4 * 2)
