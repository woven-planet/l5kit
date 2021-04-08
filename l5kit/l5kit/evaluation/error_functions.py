from typing import Callable

import torch


# Error function type
ErrorFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def l2_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """A function that takes pred, gt tensor and computes their L2 distance.

    :param pred: predicted tensor, size: [batch_size, num_dims]
    :param gt: gt tensor, size: [batch_size, num_dims]
    :return: l2 distance between the predicted and gt tensor, size: [batch_size,]
    """
    return torch.norm(pred - gt, p=2, dim=-1)
