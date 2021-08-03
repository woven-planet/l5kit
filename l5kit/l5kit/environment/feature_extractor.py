import warnings

import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models.resnet import resnet18, resnet50

from l5kit.environment import models


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor from raster images for the RL Policy.

    :param observation_space: the input observation space
    :param features_dim: the number of features to extract from the input
    :param model_arch: the model architecture used to extract the features
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256,
                 model_arch: str = "simple"):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        num_input_channels = observation_space["image"].shape[0]

        if model_arch == "simple":
            model = models.SimpleCNN(num_input_channels, features_dim)
        elif model_arch == "simpler":
            model = models.SimplerCNN(num_input_channels, features_dim)
        elif model_arch in {"resnet18", "resnet50"}:
            model = ResNetCNN(num_input_channels, features_dim, model_arch)
        elif model_arch in {"resnet1", "resnet2", "resnet3"}:
            model = CustomResNetCNN(num_input_channels, features_dim, model_arch)
        else:
            raise NotImplementedError

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = model
                total_concat_size += features_dim
            elif key == "vector":
                print("No vector attribute in observation space")
                raise NotImplementedError

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: gym.spaces.Dict) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # import pdb; pdb.set_trace()
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


def ResNetCNN(num_input_channels: int, features_dim: int, model_arch: str, pretrained: bool = True) -> nn.Module:
    """ResNet feature extractor.

    :param num_input_channels: the number of input channels in the input
    :param features_dim: the number of features to extract from input
    :param model_arch: the architecture of resnet model
    :param pretrained: flag to indicate the use of a pretrained model
    """
    if pretrained and num_input_channels != 3:
        warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

    if model_arch == "resnet18":
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(in_features=512, out_features=features_dim)
    elif model_arch == "resnet50":
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=features_dim)
    else:
        raise NotImplementedError(f"Model arch {model_arch} unknown")

    if num_input_channels != 3:
        model.conv1 = nn.Conv2d(
            num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

    return model


def CustomResNetCNN(num_input_channels: int, features_dim: int, model_arch: str) -> nn.Module:
    """Custom ResNet feature extractor.

    :param num_input_channels: the number of input channels in the input
    :param features_dim: the number of features to extract from input
    :param model_arch: the architecture of resnet model
    """

    if model_arch == "resnet1":
        model = models.CustomResNet(models.BasicBlock, [2])
        model.fc = nn.Linear(in_features=3136, out_features=features_dim)
    elif model_arch == "resnet2":
        model = models.CustomResNet(models.BasicBlock, [2, 2])
        model.fc = nn.Linear(in_features=1568, out_features=features_dim)
    elif model_arch == "resnet3":
        model = models.CustomResNet(models.BasicBlock, [2, 2, 2])
        model.fc = nn.Linear(in_features=576, out_features=features_dim)
    else:
        raise NotImplementedError(f"Model arch {model_arch} unknown")

    if num_input_channels != 3:
        model.conv = nn.Conv2d(
            num_input_channels, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )

    return model
