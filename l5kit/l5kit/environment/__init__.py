from .gym_envs import L5RasterBaseEnv, L5RasterCacheEnv, L5DatasetCacheEnv
from .gym_envs import L5Env
from .cle_utils import SimulationOutputGym

import gym
## L5 Raster rendering Env
gym.envs.register(
    id='Base-v1',
    entry_point="l5kit.environment.envs:L5RasterBaseEnvV1",
)

## Full L5 Env
gym.envs.register(
    id='L5-v0',
    entry_point="l5kit.environment.envs:L5Env",
)
