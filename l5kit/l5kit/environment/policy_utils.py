import pickle
from typing import List, Optional

import gym
from stable_baselines3.common.base_class import BaseAlgorithm

from l5kit.environment.envs.l5_env import EpisodeOutputGym, L5Env


def rollout_scene(model: BaseAlgorithm, env: gym.Env, idx: int) -> EpisodeOutputGym:
    """ Rollout a particular scene index and return the simulation output.
    :param model: the policy that predicts actions during the rollout
    :param env: the gym environment
    :param idx: the scene index to be rolled out
    :return: the episode output of the rolled out scene
    """

    # Assert
    assert isinstance(env, L5Env), "Eval environment should be an instance of L5Env"
    assert 'reset_scene_id' in env.__dict__.keys(), "Missing attribute 'reset_scene_id'"

    # Set the reset_scene_id to 'idx'
    env.reset_scene_id = idx
    obs = env.reset()
    for i in range(350):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        if done:
            break

    sim_out: EpisodeOutputGym
    sim_out = info["sim_outs"][0]
    return sim_out


def rollout_multiple_scenes(model: BaseAlgorithm, env: gym.Env, indices: List[int],
                            save_path: Optional[str] = None) -> List[EpisodeOutputGym]:
    """ Rollout multiple scenes and return the list of simulation outputs, and optionally save them.
    :param model: the policy that predicts actions during the rollout
    :param env: the gym environment
    :param indices: the scene indices to be rolled out
    :return: the list of episode outputs of the rolled out scenes
    """
    sim_outs: List[EpisodeOutputGym] = []

    for idx in indices:
        sim_out = rollout_scene(model, env, idx)
        sim_outs.append(sim_out)

    if save_path is not None:
        path = save_path + "_test_s{}_e{}".format(indices[0], indices[-1])
        with open(path + ".pkl", 'wb') as f:
            pickle.dump(sim_outs, f)

    return sim_outs
