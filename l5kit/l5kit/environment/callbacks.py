import gym
from stable_baselines3.common.callbacks import EvalCallback

from l5kit.cle.validators import ValidationCountingAggregator
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
            # Evaluate episode outputs
            self.evaluate_scenes()

            # Aggregate
            validation_results = self.metric_set.evaluator.validation_results()
            agg = ValidationCountingAggregator().aggregate(validation_results)

            # Add to current Logger
            assert self.logger is not None
            for k, v in agg.items():
                self.logger.record(f'eval/{k}', v.item())

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # reset
            self.metric_set.reset()

        return True

    def evaluate_scenes(self) -> None:
        """Evaluate the episode outputs for `n_eval_episodes` episodes.
        """
        assert self.model is not None

        obs = self.eval_env.reset()
        episodes_done = 0
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.eval_env.step(action)

            for idx in range(self.n_eval_envs):
                if done[idx]:
                    episodes_done += 1
                    self.metric_set.evaluate(info[idx]["sim_outs"])

                    if episodes_done == self.n_eval_episodes:
                        return
