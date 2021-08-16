import gym


# Register L5 Env
gym.envs.register(
    id='L5-CLE-v0',
    entry_point="l5kit.environment.envs.l5_env:L5Env",
)
