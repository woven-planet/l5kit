from typing import List, Optional

import gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback


def get_callback_list(eval_env: gym.Env, output_prefix: str, n_envs: int,
                      save_freq: int = 50000, save_path: str = './logs/',
                      eval_freq: int = 50000, n_eval_episodes: int = 10,
                      ckpt_prefix: Optional[str] = None) -> CallbackList:
    """Generate the callback list to be used during model training in L5Kit gym.
    Note: When using multiple environments, each call to ``env.step()``
    will effectively correspond to ``n_envs`` steps.
    To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param output_prefix: the prefix to save the model outputs during training
    :param n_envs: the number of parallel environments being used
    :param save_freq: the frequency to save the model
    :param save_path: the folder to save the model
    :param eval_freq: the frequency to evaluate the model
    :param n_eval_episodes: the number of episodes to evaluate
    :param ckpt_prefix: the prefix to save the model during training
    """
    callback_list: List[BaseCallback] = []

    # Save Model Periodically
    ckpt_prefix = ckpt_prefix if ckpt_prefix is not None else output_prefix
    checkpoint_callback = CheckpointCallback(save_freq=(save_freq // n_envs), save_path=save_path,
                                             name_prefix=ckpt_prefix)
    callback_list.append(checkpoint_callback)

    # Eval Model Periodically
    eval_callback = EvalCallback(eval_env, eval_freq=(eval_freq // n_envs), n_eval_episodes=n_eval_episodes)
    callback_list.append(eval_callback)

    callback = CallbackList(callback_list)
    return callback
