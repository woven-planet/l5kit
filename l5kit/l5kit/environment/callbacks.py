import math
from typing import List, Optional

import gym
from stable_baselines3.common.callbacks import EvalCallback

from l5kit.cle.metric_set import L5MetricSet
from l5kit.cle.scene_type_agg import compute_cle_scene_type_aggregations
from l5kit.cle.validators import ValidationCountingAggregator, ValidationFailedFramesAggregator
from l5kit.environment.gym_metric_set import CLEMetricSet


class L5KitEvalCallback(EvalCallback):
    """Callback for evaluating an agent using L5Kit evaluation metrics.

    :param eval_env: The environment used for initialization
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param metric_set: computes a set of metric parametrization for the L5Kit environment
    :param verbose:
    """

    def __init__(self, eval_env: gym.Env, eval_freq: int = 10000, n_eval_episodes: int = 10,
                 n_eval_envs: int = 4, metric_set: Optional[L5MetricSet] = None,
                 enable_scene_type_aggregation: Optional[bool] = True, scene_id_to_type_path: Optional[str] = None,
                 verbose: int = 0) -> None:
        super(L5KitEvalCallback, self).__init__(eval_env)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.verbose = verbose
        self.metric_set = metric_set or CLEMetricSet()

        # For scene type-based aggregation
        self.enable_scene_type_aggregation = enable_scene_type_aggregation
        self.scene_ids_to_scene_types = self._get_scene_types(scene_id_to_type_path)

    def _init_callback(self) -> None:
        pass

    def _get_scene_types(self, path: Optional[str] = None) -> List[List[str]]:
        """Construct a list mapping scene types to their corresponding types.

        :param path: Path to file contain the mapping from scene_id to scene tags
        :return: list of scene type tags per scene
        """
        scene_ids_to_scene_types = [["accn"]] * (self.n_eval_episodes // 2) + \
                                   [["dccn"]] * (self.n_eval_episodes // 2)
        return scene_ids_to_scene_types

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate episode outputs
            self.evaluate_scenes()

            # Aggregate
            validation_results = self.metric_set.evaluator.validation_results()
            # agg = ValidationCountingAggregator().aggregate(validation_results)
            agg = ValidationFailedFramesAggregator().aggregate(validation_results)

            # Add to current Logger
            assert self.logger is not None
            for k, v in agg.items():
                # print(k, v)
                # self.logger.record(f'eval/{k}', v.item())
                self.logger.record(f'eval/{k}', len(v))

            # If we should compute the scene-type aggregation metrics
            if self.enable_scene_type_aggregation:
                print("Aggregating scene types")
                # main_mset = cast(DriveFormerCLEMetricSet, self.get_metric_set(self.scene_type_aggregation_metric_prefix))
                # list_validator_tables = ["distance_validator", "collision_validator"]
                scene_type_results, scene_type_results_dict = \
                    compute_cle_scene_type_aggregations(self.metric_set,
                                                        self.scene_ids_to_scene_types,
                                                        list_validator_table_to_publish=[])
                for k, v in scene_type_results.items():
                    # print(k, v)
                    self.logger.record(f'{k}', v)

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

        self._set_reset_ids()
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

    def _set_reset_ids(self) -> None:
        """Reset scene_ids for deterministic unroll"""
        reset_interval = math.ceil(self.n_eval_episodes / self.n_eval_envs)
        reset_indices = [reset_interval * i for i in range(self.n_eval_envs)]
        for idx in range(self.n_eval_envs):
            self.eval_env.env_method("set_reset_id", reset_indices[idx], indices=[idx])
        return

# import os
# os.environ["L5KIT_DATA_FOLDER"] = os.environ["HOME"] + '/level5_data/'
# env = gym.make("L5-CLE-v0", env_config_path='../../../examples/RL/gym_config.yaml')
# test_callback = L5KitEvalCallback(env)