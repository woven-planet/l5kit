import os
import pickle
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from l5kit.rasterization import Rasterizer


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


def visualize_input_raster(rasterizer: Rasterizer, image: torch.Tensor) -> None:
    """Visualize the input raster image

    :param rasterizer: the rasterizer
    :param image: numpy array
    :return: the numpy dict with 'positions' and 'yaws'
    """

    image = image.permute(1, 2, 0).cpu().numpy()
    output_im = rasterizer.to_rgb(image)

    im = Image.fromarray(output_im)

    i = 0
    while os.path.exists("imgs/test%s.png" % i):
        i += 1
    print("Saving Raster RGB Image")
    im.save("imgs/test%s.png" % i)


def error_stats(path: str) -> None:
    """L2 error between groundtruth and prediction during Open loop training

    :param path: the path to pkl file containing the groundtruth and predictions
    """
    with open(path + ".pkl", 'rb') as f:
        [gt_action_list, action_list] = pickle.load(f)
        error = gt_action_list - action_list
        error = np.linalg.norm(error, axis=1)
        print("Error: ", error)
    return
