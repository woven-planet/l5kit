import csv
from pathlib import Path
from typing import Dict, List, NamedTuple

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


def get_scene_types(scene_id_to_type_path: str) -> List[List[str]]:
    """Construct a list mapping scene ids to their corresponding types.

    :param scene_id_to_type_path: path to the mapping.
    :return: list of scene type tags per scene
    """
    # Read csv
    scene_type_dict: Dict[int, str]
    with open(scene_id_to_type_path, 'r') as f:
        csv_reader = csv.reader(f)
        scene_type_dict = {int(rows[0]): rows[1] for rows in csv_reader}

    # Convert dict to List[List[str]]
    scene_id_to_type_list: List[List[str]] = []
    for k, v in scene_type_dict.items():
        scene_id_to_type_list.append([v])
    return scene_id_to_type_list


def get_scene_types_as_dict(scene_id_to_type_path: str) -> Dict[str, List[int]]:
    """Construct a list mapping scene ids to their corresponding types.

    :param scene_id_to_type_path: path to the mapping.
    :return: dict of scene types and the corresponding scene_id list
    """
    # Read csv
    scene_type_dict: Dict[int, str]
    with open(scene_id_to_type_path, 'r') as f:
        csv_reader = csv.reader(f)
        scene_type_dict = {int(rows[0]): rows[1] for rows in csv_reader}

    # Convert dict to List[List[str]]
    scene_id_to_type_dict: Dict[str, List[int]] = {}
    for k, v in scene_type_dict.items():
        if v not in scene_id_to_type_dict:
            scene_id_to_type_dict[v] = [k]
        else:
            scene_id_to_type_dict[v].append(k)
    return scene_id_to_type_dict
