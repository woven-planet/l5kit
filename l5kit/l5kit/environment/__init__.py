import gym

from .cle_utils import SimulationOutputGym
from .feature_extractor import ResNetCNN

# L5 Raster rendering Env
gym.envs.register(
    id='Base-v1',
    entry_point="l5kit.environment.envs:L5RasterBaseEnvV1",
)

# Full L5 Env
gym.envs.register(
    id='L5-v0',
    entry_point="l5kit.environment.envs:L5Env",
)
