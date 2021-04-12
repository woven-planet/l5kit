import unittest

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
