from typing import Callable

import numpy as np


class LambdaRandomGenerator:
    """
    LambdaRandomGenerator generates values by calling the supplied function.
    Note that instead of this you could also just use this supplied function itself.
    """

    def __init__(self, sampling_function: Callable[[], np.ndarray]):
        self.sampling_function = sampling_function

    def _sample(self) -> np.ndarray:
        return self.sampling_function()

    def __call__(self) -> np.ndarray:
        return self._sample()


class ReplayRandomGenerator:
    """
    ReplayRandomGenerator generates values by sequentially returning the values in the supplied
    array. It does not loop or reset automatically.
    """

    def __init__(self, values: np.ndarray):
        self._values = values
        self._idx = 0

    def _sample(self) -> np.ndarray:
        if self._idx == len(self._values):
            raise IndexError("ReplayRandomGenerator is out of values for index={}".format(self._idx))
        val = np.array(self._values[self._idx])
        self._idx += 1
        return val

    def __call__(self) -> np.ndarray:
        return self._sample()


class GaussianRandomGenerator:
    """
    GaussianRandomGenerator generates values by sampling from a normal distribution with specified
    mean and standard deviation. Note that this gaussian can be multidimensional.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def _sample(self) -> np.ndarray:
        return np.random.normal(self.mean, self.std)

    def __call__(self) -> np.ndarray:
        return self._sample()


class UniformRandomGenerator:
    """
    GaussianRandomGenerator generates values by sampling from a normal distribution with specified
    mean and standard deviation. Note that this gaussian can be multidimensional.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = low
        self.high = high

    def _sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high)

    def __call__(self) -> np.ndarray:
        return self._sample()
