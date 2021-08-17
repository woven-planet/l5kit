import math
from pathlib import Path
from typing import NamedTuple

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
    steer_scale: float = math.radians(20) * 0.1  # 15, 20, 30
    acc_scale: float = 0.6  # 0.3, 0.6, 0.6


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
    assert x_mu - 10 * x_std <= min(x_components)
    assert x_mu + 10 * x_std >= max(x_components)

    y_mu, y_std = np.mean(y_components), np.std(y_components)
    assert y_mu - 10 * y_std <= min(y_components)
    assert y_mu + 10 * y_std >= max(y_components)

    yaw_mu, yaw_std = np.mean(yaw_components), np.std(yaw_components)
    assert yaw_mu - 10 * yaw_std <= min(yaw_components)
    assert yaw_mu + 10 * yaw_std >= max(yaw_components)

    # Keeping scale = 10 * std so that extreme values are not clipped
    return NonKinematicActionRescaleParams(x_mu, 10 * x_std, y_mu, 10 * y_std, yaw_mu, 10 * yaw_std)


def calculate_kinematic_rescale_params(sim_dataset: SimulationDataset) -> KinematicActionRescaleParams:
    """Calculate the action un-normalization parameters from the simulation dataset for kinematic model.

    :param sim_dataset: the input dataset to calculate the action rescale parameters
    :return: the unnormalized action
    """
    # v_component_frames = []
    # yaw_component_frames = []

    # for index in range(1, len(sim_dataset)):
    #     ego_input = sim_dataset.rasterise_frame_batch(index)
    #     v_component_frames.append([scene['curr_speed'].item() for scene in ego_input])
    #     yaw_component_frames.append([scene['target_yaws'][0, 0] for scene in ego_input])

    # v_components = np.stack(v_component_frames)
    # acc_components = v_components[1:] - v_components[:-1]
    # acc_components = 0.5 * (acc_components[1:] + acc_components[:-1])
    # acc_components = 0.5 * (acc_components[1:] + acc_components[:-1])
    # acc_components = 0.5 * (acc_components[1:] + acc_components[:-1])
    # acc_components = acc_components.flatten()
    # yaw_components = np.concatenate(yaw_component_frames)
    # print(max(acc_components), min(acc_components))
    # print(max(yaw_components), min(yaw_components), math.radians(20) * 0.1)
    # import pdb; pdb.set_trace()

    # Straight; 64 time steps
    # return KinematicActionRescaleParams(math.radians(15) * 0.1, 0.3)
    # Turn; 64 time steps
    # return KinematicActionRescaleParams(math.radians(30) * 0.1, 0.3)

    # Overfit [10]: (~) 25, 0.3, [100]: 20, 0.6, [1000]: 30, 0.6
    return KinematicActionRescaleParams(math.radians(30) * 0.1, 0.6)


def save_input_raster(rasterizer: Rasterizer, image: torch.Tensor, num_images: int = 20,
                      output_folder: str = 'raster_inputs') -> None:
    """Save the input raster image.

    :param rasterizer: the rasterizer
    :param image: numpy array
    :param num_images: number of images to save
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
    img_path = output_folder / f"input{i}.png"
    while img_path.exists():
        i += 1
        img_path = output_folder / f"input{i}.png"

    # save
    im.save(img_path)

    # exit code once num_images images saved
    if i == num_images:
        exit()
