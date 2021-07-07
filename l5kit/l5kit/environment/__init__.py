import gym

from .cle_utils import SimulationOutputGym
from .feature_extractor import ResNetCNN
from .gym_envs import L5DatasetCacheEnv, L5Env, L5RasterBaseEnv, L5RasterCacheEnv


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
