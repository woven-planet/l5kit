import math
import numbers
from pathlib import Path
from typing import Any, Dict, NamedTuple

import numpy as np
import torch
from PIL import Image

from l5kit.rasterization import Rasterizer
from l5kit.simulation.dataset import SimulationDataset


class NonKinematicActionRescaleParams(NamedTuple):
    """Defines the parameters to rescale actions into un-normalized action space
    when the kinematic model is not used.

    :param x_mu: the translation of the x-coordinate
    :param x_scale: the scaling of the x-coordinate
    :param y_mu: the translation of the y-coordinate
    :param y_scale: the scaling of the y-coordinate
    :param yaw_mu: the translation of the yaw (radians)
    :param yaw_scale: the scaling of the yaw (radians)
    """
    x_mu: float
    x_scale: float
    y_mu: float
    y_scale: float
    yaw_mu: float
    yaw_scale: float


class KinematicActionRescaleParams(NamedTuple):
    """Defines the parameters to rescale actions into un-normalized action space
    when the kinematic model is used.

    :param steer_scale: the scaling of the steer (kinematic model)
    :param acc_scale: the scaling of the acceleration (kinematic model)
    """
    steer_scale: float
    acc_scale: float


def calculate_non_kinematic_rescale_params(sim_dataset: SimulationDataset) -> NonKinematicActionRescaleParams:
    """Calculate the action un-normalization parameters from the simulation dataset for non-kinematic model.

    :param sim_dataset: the input dataset to calculate the action rescale parameters
    :return: the unnormalized action
    """
    x_component_frames = []
    y_component_frames = []
    yaw_component_frames = []

    for index in range(1, len(sim_dataset) - 1):
        ego_input = sim_dataset.rasterise_frame_batch(index)
        x_component_frames.append([scene['target_positions'][0, 0] for scene in ego_input])
        y_component_frames.append([scene['target_positions'][0, 1] for scene in ego_input])
        yaw_component_frames.append([scene['target_yaws'][0, 0] for scene in ego_input])

    x_components = np.concatenate(x_component_frames)
    y_components = np.concatenate(y_component_frames)
    yaw_components = np.concatenate(yaw_component_frames)

    x_mu, x_std = np.mean(x_components), np.std(x_components)
    y_mu, y_std = np.mean(y_components), np.std(y_components)
    yaw_mu, yaw_std = np.mean(yaw_components), np.std(yaw_components)

    # Keeping scale = 10 * std so that extreme values are not clipped
    return NonKinematicActionRescaleParams(x_mu, 10 * x_std, y_mu, 10 * y_std, yaw_mu, 10 * yaw_std)


def calculate_kinematic_rescale_params(sim_dataset: SimulationDataset) -> KinematicActionRescaleParams:
    """Calculate the action un-normalization parameters from the simulation dataset for kinematic model.

    :param sim_dataset: the input dataset to calculate the action rescale parameters
    :return: the unnormalized action
    """
    return KinematicActionRescaleParams(math.radians(20) * 0.1, 0.6)


def convert_to_numpy(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a dict into numpy dict (on cpu).

    :param data: the dict with both torch and numpy entries
    :return: the numpy dict
    """
    output_data = {}
    for k, v in data.items():
        if isinstance(v, numbers.Number):
            output_data[k] = np.array([v])
        elif isinstance(v, np.ndarray):
            output_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, torch.Tensor):
            output_data[k] = np.expand_dims(v.cpu().numpy(), axis=0)
        else:
            raise NotImplementedError(f"{type(v)} is not supported (field {k})")
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
