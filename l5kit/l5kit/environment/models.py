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


def AlexCNN_GN(num_input_channels: int, features_dim: int) -> nn.Module:
    """A simplified feature extractor with GroupNorm.

    :param num_input_channels: the number of input channels in the input
    :param features_dim: the number of features to extract from input
    """
    model = nn.Sequential(
        # Conv1
        nn.Conv2d(num_input_channels, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.GroupNorm(6, 96),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Conv2
        nn.Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.GroupNorm(8, 256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Conv3
        nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.GroupNorm(8, 384),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Conv4
        nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.GroupNorm(8, 384),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Conv5
        nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.GroupNorm(8, 256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=2304, out_features=features_dim),
    )

    return model
