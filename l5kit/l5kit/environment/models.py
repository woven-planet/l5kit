import torch.nn as nn


def SimpleCNN_GN(num_input_channels: int, features_dim: int) -> nn.Module:
    """A simplified feature extractor with GroupNorm.

    :param num_input_channels: the number of input channels in the input
    :param features_dim: the number of features to extract from input
    """
    model = nn.Sequential(
        nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.GroupNorm(4, 64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.GroupNorm(2, 32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=1568, out_features=features_dim),
    )

    return model
