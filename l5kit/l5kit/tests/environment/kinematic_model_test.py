import unittest

import numpy as np
import pytest

from l5kit.environment import kinematic_model


class TestUnicycleModel(unittest.TestCase):
    def test_model_reset(self) -> None:
        kin_model = kinematic_model.UnicycleModel()
        init_state = np.array([0., 0., 0., 2.0])
        kin_model.reset(init_state)

        x, y, r, v = kin_model.old_x, kin_model.old_y, kin_model.old_r, kin_model.old_v
        self.assertListEqual([x, y, r, v], [0., 0., 0., 2.], 4)

    def test_model_constraints(self) -> None:
        kin_model = kinematic_model.UnicycleModel()
        init_state = np.array([0., 0., 0., 2.0])
        kin_model.reset(init_state)

        input_action = np.array([0.02, 3.0])
        with pytest.raises(AssertionError):
            kin_model.update(input_action)

        input_action = np.array([0.1, 0.0])
        with pytest.raises(AssertionError):
            kin_model.update(input_action)

    def test_model_update(self) -> None:
        kin_model = kinematic_model.UnicycleModel()
        init_state = np.array([0., 0., 0., 2.0])
        kin_model.reset(init_state)

        input_action = np.array([0.01, 0.1])
        kin_model.update(input_action)
        x, y, r, v = kin_model.old_x, kin_model.old_y, kin_model.old_r, kin_model.old_v
        self.assertListEqual([x, y, r, v], [0., 0., 0., 2.1], 4)

        input_action = np.array([-0.01, -0.3])
        kin_model.update(input_action)
        x, y, r, v = kin_model.old_x, kin_model.old_y, kin_model.old_r, kin_model.old_v
        self.assertListEqual([x, y, r, v], [0., 0., 0., 1.8], 4)

        kin_model.reset(init_state)
        x, y, r, v = kin_model.old_x, kin_model.old_y, kin_model.old_r, kin_model.old_v
        self.assertListEqual([x, y, r, v], [0., 0., 0., 2.], 4)
