import unittest

import numpy as np

from l5kit.kinematic import fit_ackerman_model_approximate, fit_ackerman_model_exact


class TestAckermanSteeringModel(unittest.TestCase):
    def test_fit_ackerman_steering(self) -> None:
        # These are only smoke tests for now, to be improved
        test_trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        test_velocity = np.ones_like(test_trajectory)
        w = np.ones(len(test_trajectory))

        x, y, r, v = fit_ackerman_model_approximate(
            test_trajectory[:, 0],
            test_trajectory[:, 1],
            test_velocity[:, 0],
            test_velocity[:, 1],
            w,
            w,
            w,
            w,
            w,
            w,
            w,
            w,
        )

        N = len(test_trajectory)
        self.assertEqual(x.shape, (N,))
        self.assertEqual(y.shape, (N,))
        self.assertEqual(r.shape, (N,))
        self.assertEqual(v.shape, (N,))

    def test_fit_ackerman_steering_exact(self) -> None:
        # These are only smoke tests for now, to be improved
        test_trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        test_velocity = np.ones(len(test_trajectory))
        test_angle = np.zeros(len(test_trajectory))
        w = np.ones(len(test_trajectory))
        x, y, r, v, acc, steer = fit_ackerman_model_exact(
            test_trajectory[:, 0], test_trajectory[:, 1], test_angle, test_velocity, w, w, w, w, w, w, w, w
        )

        N = len(test_trajectory)
        self.assertEqual(x.shape, (N,))
        self.assertEqual(y.shape, (N,))
        self.assertEqual(r.shape, (N,))
        self.assertEqual(v.shape, (N,))


if __name__ == "__main__":
    unittest.main()
