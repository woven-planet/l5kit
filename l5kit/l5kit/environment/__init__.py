import gym

# Register L5 Env
gym.envs.register(
    id='L5cle-v0',
    entry_point="l5kit.environment.envs:L5Env",
)
