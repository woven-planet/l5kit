from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from l5kit.rasterization import Rasterizer

def rescale_action(action, x_mu=1.20, x_scale=0.2, y_mu=0.0, y_scale=0.03, yaw_scale=3.14):
    assert len(action) == 3
    action[0] = x_mu + x_scale * action[0]
    action[1] = y_mu + y_scale * action[1]
    action[2] = yaw_scale * action[2]
    return action


def default_collate_numpy(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Move a torch dict into numpy (on cpu)

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


def convert_to_dict(data: np.ndarray, future_num_frames: int) -> Dict[str, np.ndarray]:
    """Convert vector into numpy dict

    :param data: numpy array
    :param future_num_frames: number of frames predicted
    :return: the numpy dict with 'positions' and 'yaws'
    """
    # [batch_size=1, num_steps, (X, Y, yaw)]
    data = data.reshape(1, future_num_frames, 3)
    pred_positions = data[:, :, :2]
    # [batch_size, num_steps, 1->(yaw)]
    pred_yaws = data[:, :, 2:3]
    data_dict = {"positions": pred_positions, "yaws": pred_yaws}
    return data_dict


def visualize_input_raster(rasterizer: Rasterizer, image: torch.Tensor,
                           output_folder: Optional[str] = 'raster_inputs') -> None:
    """Visualize the input raster image

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
