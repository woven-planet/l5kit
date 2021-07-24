from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from l5kit.rasterization import Rasterizer


def default_collate_numpy(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Move a torch dict into numpy (on cpu).

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


def visualize_input_raster(rasterizer: Rasterizer, image: torch.Tensor,
                           output_folder: str = 'raster_inputs') -> None:
    """Visualize the input raster image.

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
