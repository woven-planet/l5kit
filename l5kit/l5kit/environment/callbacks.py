from typing import List

import gym
from stable_baselines3.common.callbacks import EvalCallback

from l5kit.cle.validators import ValidationCountingAggregator
from l5kit.environment.envs.l5_env import EpisodeOutputGym
from l5kit.environment.gym_metric_set import DisplacementCollisionMetricSet


class L5KitEvalCallback(EvalCallback):
    """Callback for evaluating an agent using L5Kit evaluation metrics.

    :param eval_env: The environment used for initialization
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param verbose:
    :param metric_set: computes a set of metric parametrization for the L5Kit environment
    """

    def __init__(self, eval_env: gym.Env, eval_freq: int = 10000, n_eval_episodes: int = 10,
                 n_eval_envs: int = 4, verbose: int = 0) -> None:
        super(L5KitEvalCallback, self).__init__(eval_env)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.verbose = verbose
        self.metric_set = DisplacementCollisionMetricSet()

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Get episode outputs
            episode_outputs = self.get_eval_info()

            # Evaluate
            self.metric_set.evaluator.evaluate(episode_outputs)
            validation_results = self.metric_set.evaluator.validation_results()
            _ = ValidationCountingAggregator().aggregate(validation_results)

            # TODO Log the aggregated results

            # # Add to current Logger
            # self.logger.record("eval/mean_reward", float(mean_reward))
            # self.logger.record("eval/mean_ep_length", mean_ep_length)

            # # Dump log so the evaluation results are printed with the correct timestep
            # self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
            # self.logger.dump(self.num_timesteps)

        return True

    def get_eval_info(self) -> List[EpisodeOutputGym]:
        """Callback the episode outputs for `n_eval_episodes` episodes.

        :return: the list of episode outputs
        """
        assert self.model is not None

        obs = self.eval_env.reset()
        episodes_done = 0
        episode_outputs: List[EpisodeOutputGym] = []
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.eval_env.step(action)

            for idx in range(self.n_eval_envs):
                if done[idx]:
                    episodes_done += 1
                    episode_outputs.append(info[idx]["sim_outs"][0])

                    if episodes_done == self.n_eval_episodes:
                        return episode_outputs
