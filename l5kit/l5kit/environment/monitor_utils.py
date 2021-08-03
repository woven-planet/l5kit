from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import pandas as pd
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import load_results, Monitor


def monitor_env(env: gym.Env, monitor_dir: str, monitor_kwargs: Optional[Dict[str, Any]] = None) -> gym.Env:
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: the environment with the monitor wrapper
    """
    # Wrap the env in a Monitor wrapper
    # to have additional training information
    monitor_dir = Path(monitor_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = str(monitor_dir / str(0))

    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    env = Monitor(env, filename=monitor_path, **monitor_kwargs)
    return env


def ts2xy(data_frame: pd.DataFrame, info_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: the input data
    :param info_key: the info keyword to plot
    :return: the x and y output
    """
    x_var = np.cumsum(data_frame.l.values)
    assert info_key in data_frame, KeyError
    y_var = data_frame.get(info_key).values
    return x_var, y_var


def plot_results(dirs: List[str], num_timesteps: Optional[int], info_key: str,
                 figsize: Tuple[int, int] = (8, 2)) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param info_key: the info keyword to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, info_key) for data_frame in data_frames]
    results_plotter.plot_curves(xy_list, results_plotter.X_TIMESTEPS, info_key, figsize)
