from typing import Callable

import numpy as np
import pytest

from l5kit.random import GaussianRandomGenerator, LambdaRandomGenerator, ReplayRandomGenerator


def test_replay() -> None:
    # One-dimensional
    log = np.arange(0, 10)
    r = ReplayRandomGenerator(log)
    for idx in range(10):
        val = r()
        assert np.all(val == np.array(log[idx]))

    # The log ran out at this point, so should throw
    with pytest.raises(Exception):
        r()


def test_replay_multidim() -> None:
    log = np.array([[0, 1], [2, 3]])
    r = ReplayRandomGenerator(log)
    assert np.all(r() == np.array([0, 1]))
    assert np.all(r() == np.array([2, 3]))
    with pytest.raises(Exception):
        r()


def test_lambda_simple() -> None:
    def f() -> np.ndarray:
        return np.array(5)  # I rolled a dice for this

    r = LambdaRandomGenerator(f)
    assert np.all(r() == np.array(5))
    assert np.all(r() == np.array(5))


def test_lambda_closure() -> None:
    # Lambda with some state in a closure
    def get_f() -> Callable:
        i = np.array(0)

        def f() -> np.ndarray:
            nonlocal i
            i += 1
            return i

        return f

    r = LambdaRandomGenerator(get_f())

    assert np.all(r() == np.array(1))
    assert np.all(r() == np.array(2))


def test_2d_gaussian_without_std() -> None:
    mean = np.array([0, 5])
    r = GaussianRandomGenerator(mean=mean, std=np.zeros(2))

    assert np.all(r() == mean)
