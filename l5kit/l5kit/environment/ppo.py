import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.environment.feature_extractor import ResNetCNN


use_subProc = True

# Custom Feature Extractor with ResNet backbone
policy_kwargs = dict(
    features_extractor_class=ResNetCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

# make gym env
# env = gym.make("L5-v0")

# custom wrap env into VecEnv
if use_subProc:
    env = make_vec_env("L5-v0", n_envs=2, vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=dict(start_method='fork'))

# define model
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=128)
print("Model created.....")

# train
model.learn(300)
print("Done")
