import math
from pathlib import Path
from typing import Any, Dict, NamedTuple

import numpy as np
import torch
from PIL import Image

from l5kit.rasterization import Rasterizer
from l5kit.simulation.dataset import SimulationDataset


class ActionRescaleParams(NamedTuple):
    """Defines the parameters to rescale model actions into un-normalized action space.

    :param x_mu: the translation of the x-coordinate
    :param x_scale: the scaling of the x-coordinate
    :param y_mu: the translation of the y-coordinate
    :param y_scale: the scaling of the y-coordinate
    :param yaw_mu: the translation of the yaw (radians)
    :param yaw_scale: the scaling of the yaw (radians)
    :param steer_scale: the scaling of the steer (kinematic model)
    :param acc_scale: the scaling of the acceleration (kinematic model)

    """
    x_mu: float = 1.20
    x_scale: float = 0.20
    y_mu: float = 0.0
    y_scale: float = 0.03
    yaw_mu: float = 0.0
    yaw_scale: float = 0.3
    steer_scale: float = math.radians(15) * 0.1
    acc_scale: float = 0.3


def calculate_rescale_params(sim_dataset: SimulationDataset, use_kinematic: bool = False) -> ActionRescaleParams:
    """Calculate the action un-normalization parameters from the simulation dataset.

    :param sim_dataset: the input dataset to calculate the action rescale parameters
    :param use_kinematic: flag to use the kinematic model
    :return: the unnormalized action
    """
    if use_kinematic:
        v_component_frames = []
        yaw_component_frames = []

        for index_ in range(1, len(sim_dataset)):
            ego_input = sim_dataset.rasterise_frame_batch(index_)
            v_component_frames.append([scene['curr_speed'].item() for scene in ego_input])
            yaw_component_frames.append([scene['target_yaws'][0, 0] for scene in ego_input])

        v_components = np.stack(v_component_frames)
        acc_components = v_components[1:] - v_components[:-1]
        acc_components = acc_components.flatten()
        acc_mu, acc_std = np.mean(acc_components), np.std(acc_components)

        yaw_components = np.concatenate(yaw_component_frames)
        yaw_mu, yaw_std = np.mean(yaw_components), np.std(yaw_components)

        v_components = np.stack(v_component_frames)

        assert max(acc_components) <= 0.7
        assert -0.7 <= min(acc_components)
        return ActionRescaleParams()

    x_component_frames = []
    y_component_frames = []
    yaw_component_frames = []

    for index_ in range(1, len(sim_dataset) - 1):
        ego_input = sim_dataset.rasterise_frame_batch(index_)
        x_component_frames.append([scene['target_positions'][0, 0] for scene in ego_input])
        y_component_frames.append([scene['target_positions'][0, 1] for scene in ego_input])
        yaw_component_frames.append([scene['target_yaws'][0, 0] for scene in ego_input])

    x_components = np.concatenate(x_component_frames)
    y_components = np.concatenate(y_component_frames)
    yaw_components = np.concatenate(yaw_component_frames)

    x_mu, x_std = np.mean(x_components), np.std(x_components)
    assert x_mu - 10 * x_std <= min(x_components)
    assert x_mu + 10 * x_std >= max(x_components)

    y_mu, y_std = np.mean(y_components), np.std(y_components)
    assert y_mu - 10 * y_std <= min(y_components)
    assert y_mu + 10 * y_std >= max(y_components)

    yaw_mu, yaw_std = np.mean(yaw_components), np.std(yaw_components)
    assert yaw_mu - 10 * yaw_std <= min(yaw_components)
    assert yaw_mu + 10 * yaw_std >= max(yaw_components)

    return ActionRescaleParams(x_mu, 10 * x_std, y_mu, 10 * y_std, yaw_mu, 10 * yaw_std)


def convert_to_numpy(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a dict into numpy dict (on cpu).

    :param data: the dict with both torch and numpy entries
    :return: the numpy dict
    """
    output_data = {}
    for k, v in data.items():
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, np.int64) or isinstance(v, np.float32):
            output_data[k] = np.array([v])
        elif isinstance(v, np.ndarray):
            output_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, torch.Tensor):
            output_data[k] = np.expand_dims(v.cpu().numpy(), axis=0)
        else:
            print(k, v, type(v))
            raise NotImplementedError
    return output_data


def save_input_raster(rasterizer: Rasterizer, image: torch.Tensor, output_folder: str = 'raster_inputs') -> None:
    """Save the input raster image.

    :param rasterizer: the rasterizer
    :param image: numpy array
    :param output_folder: directory to save the image
    :return: the numpy dict with 'positions' and 'yaws'
    """

    image = image.permute(1, 2, 0).cpu().numpy()
    output_im = rasterizer.to_rgb(image)

    im = Image.fromarray(output_im)

    # mkdir
    Path(output_folder).mkdir(exist_ok=True)
    output_folder = Path(output_folder)

    # loop
    i = 0
    img_path = output_folder / 'input{}.png'.format(i)
    while img_path.exists():
        i += 1
        img_path = output_folder / 'input{}.png'.format(i)

    # save
    im.save(img_path)

    # exit code once 20 images saved
    if i == 20:
        exit()
