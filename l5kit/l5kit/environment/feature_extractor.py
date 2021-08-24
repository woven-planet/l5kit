import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from l5kit.environment import models


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor from raster images for the RL Policy.

    :param observation_space: the input observation space
    :param features_dim: the number of features to extract from the input
    :param model_arch: the model architecture used to extract the features
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256,
                 model_arch: str = "simple_gn"):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        num_input_channels = observation_space["image"].shape[0]

        if model_arch == 'simple_gn':
            # A simplified feature extractor with GroupNorm.
            model = models.SimpleCNN_GN(num_input_channels, features_dim)
        else:
            raise NotImplementedError

        extractors = {"image": model}
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = features_dim

    def forward(self, observations: gym.spaces.Dict) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
