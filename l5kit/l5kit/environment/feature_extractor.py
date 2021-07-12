import warnings

import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models.resnet import resnet18, resnet50

from .utils import visualize_input_raster

class ResNetCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256,
                 model_arch: str = "resnet18", pretrained: bool = True):
        super(ResNetCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        num_input_channels = observation_space["image"].shape[0]

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

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # Raster Image
                extractors[key] = model
                total_concat_size += features_dim
            elif key == "vector":
                print("No vector attribute in observation space")
                raise NotImplementedError

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

        # Visualization
        # from l5kit.configs import load_config_data
        # from l5kit.data import LocalDataManager
        # from l5kit.rasterization import build_rasterizer
        # dm = LocalDataManager(None)
        # # get config
        # cfg = load_config_data("/home/ubuntu/src/l5kit/l5kit/l5kit/environment/envs/config.yaml")
        # # rasterisation
        # self.rasterizer = build_rasterizer(cfg, dm)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # out_im = visualize_input_raster(self.rasterizer, observations['image'][0])
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)

class SimpleCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(SimpleCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        num_input_channels = observation_space["image"].shape[0]

        print("Simple Model")
        model = torch.nn.Sequential(
                    nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                    nn.Linear(in_features=1568, out_features=features_dim),
        )

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # Raster Image
                extractors[key] = model
                total_concat_size += features_dim
            elif key == "vector":
                print("No vector attribute in observation space")
                raise NotImplementedError

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
