import unittest
from typing import Callable

import numpy as np

from l5kit.random import GaussianRandomGenerator, LambdaRandomGenerator, ReplayRandomGenerator


class TestSample(unittest.TestCase):
    def test_replay(self) -> None:
        # One-dimensional
        log = np.arange(0, 10)
        r = ReplayRandomGenerator(log)
        for idx in range(10):
            val = r()
            self.assertEqual(val, np.array(log[idx]))

        # The log ran out at this point, so should throw
        self.assertRaises(Exception, r)
        self.assertRaises(Exception, r)

    def test_replay_multidim(self) -> None:
        log = np.array([[0, 1], [2, 3]])
        r = ReplayRandomGenerator(log)
        self.assertTrue(np.array_equal(r(), np.array([0, 1])))
        self.assertTrue(np.array_equal(r(), np.array([2, 3])))
        self.assertRaises(IndexError, r)

    def test_lambda_simple(self) -> None:
        def f() -> np.ndarray:
            return np.array(5)  # I rolled a dice for this

        r = LambdaRandomGenerator(f)
        self.assertTrue(np.array_equal(r(), np.array(5)))
        self.assertTrue(np.array_equal(r(), np.array(5)))

    def test_lambda_closure(self) -> np.ndarray:
        # Lambda with some state in a closure
        def get_f() -> Callable:
            i = np.array(0)

            def f() -> np.ndarray:
                nonlocal i
                i += 1
                return i

            return f

        r = LambdaRandomGenerator(get_f())

        self.assertTrue(np.array_equal(r(), np.array(1)))
        self.assertTrue(np.array_equal(r(), np.array(2)))

    def test_2d_gaussian_without_std(self) -> None:
        mean = np.array([0, 5])
        r = GaussianRandomGenerator(mean=mean, std=np.zeros(2))

        self.assertTrue(np.array_equal(r(), mean))


if __name__ == "__main__":
    unittest.main()
