import unittest

import numpy as np

from l5kit.environment import kinematic_model


class TestUnicycleModel(unittest.TestCase):
    def test_model_reset(self) -> None:
        kin_model = kinematic_model.UnicycleModel()
        init_state = np.array([0., 0., 0., 2.0])
        kin_model.reset(init_state)

        x, y, r, v = kin_model.old_x, kin_model.old_y, kin_model.old_r, kin_model.old_v
        self.assertListEqual([x, y, r, v], [0., 0., 0., 2.], 4)

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

    def test_model_constraints(self) -> None:
        kin_model = kinematic_model.UnicycleModel(min_acc=-0.6, max_acc=0.6,
                                                  min_steer=-0.2, max_steer=0.2)
        init_state = np.array([0., 0., 0., 2.0])
        kin_model.reset(init_state)

        input_action = np.array([0.02, 3.0])
        kin_model.update(input_action)
        x, y, r, v = kin_model.old_x, kin_model.old_y, kin_model.old_r, kin_model.old_v
        self.assertListEqual([x, y, r, v], [0., 0., 0., 2.6], 4)

        input_action = np.array([1.0, 0.0])
        kin_model.update(input_action)
        self.assertEqual(kin_model.new_r, kin_model.max_steer)
