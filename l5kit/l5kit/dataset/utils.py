from typing import Dict

import numpy as np
import torch


kMaxStrLength = 70  # Max string length for fixed length byte encoding.


def convert_str_to_fixed_length_tensor(string: str, max_length: int = kMaxStrLength) -> torch.Tensor:
    """
    Converts a string into a fixed length tensor of type torch.uint8.

    Args:
        string (str): String to convert
        max_length (int): Fixed length, default kMaxStrLength

    Returns:
        torch.Tensor: A fixed length tensor of type torch.uint8 that encodes the string
    """
    assert (
        len(string) <= max_length
    ), f"Encountered string longer that maximum length supported ({max_length}): {string}"
    assert "\0" not in string, f"String contains 0 value used for padding: {string}"
    return torch.cat(
        (
            torch.ByteTensor(torch.ByteStorage.from_buffer(string.encode("ascii"))),  # type: ignore
            torch.zeros(max_length - len(string), dtype=torch.uint8),
        )
    )


def move_to_device(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move the data dict to a given torch device

    :param data: the torch dict
    :param device: the device where to move each value of the dict
    :return: the torch dict on the new device
    """
    return {k: v.to(device) for k, v in data.items()}


def move_to_numpy(data: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Move a torch dict into numpy (on cpu)

    :param data: the torch dict
    :return: the numpy dict
    """
    return {k: v.cpu().numpy() for k, v in data.items()}
