import csv
import math
from typing import Dict, List, Optional, Tuple

import gym
from stable_baselines3.common.callbacks import EvalCallback

from l5kit.cle.composite_metrics import CompositeMetricAggregator
from l5kit.cle.metric_set import L5MetricSet
from l5kit.cle.scene_type_agg import compute_cle_scene_type_aggregations
from l5kit.cle.validators import ValidationCountingAggregator
from l5kit.environment.gym_metric_set import CLEMetricSet


class L5KitEvalCallback(EvalCallback):
    """Callback for evaluating an agent using L5Kit evaluation metrics.

    :param eval_env: The environment used for initialization
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param metric_set: computes a set of metric parametrization for the L5Kit environment
    :param enable_scene_type_aggregation: enable evaluation according to scene type
    :param scene_id_to_type_path: path to the csv file mapping scene id to scene type
    :param prefix: the prefix to save the computed metrics
    :param verbose:
    """

    def __init__(self, eval_env: gym.Env, eval_freq: int = 10000, n_eval_episodes: int = 10,
                 n_eval_envs: int = 4, metric_set: Optional[L5MetricSet] = None,
                 enable_scene_type_aggregation: Optional[bool] = False, scene_id_to_type_path: Optional[str] = None,
                 prefix: str = 'eval', verbose: int = 0) -> None:
        super(L5KitEvalCallback, self).__init__(eval_env)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.verbose = verbose
        self.prefix = prefix
        self.metric_set = metric_set or CLEMetricSet()
        # Assert required metrics exist.
        if 'displacement_error_l2' not in self.metric_set.evaluation_plan.metrics_dict():
            raise RuntimeError('\'displacement_error_l2\' missing in metric set')
        if 'collision_front' not in self.metric_set.evaluation_plan.metrics_dict() or \
            'collision_rear' not in self.metric_set.evaluation_plan.metrics_dict() or \
                'collision_side' not in self.metric_set.evaluation_plan.metrics_dict():
            raise RuntimeError('\'Collision metrics\' missing in metric set')

        # For scene type-based aggregation
        self.enable_scene_type_aggregation = enable_scene_type_aggregation
        if self.enable_scene_type_aggregation:
            assert scene_id_to_type_path is not None
            self.scene_id_to_type_path = scene_id_to_type_path
            self.scene_ids_to_scene_types = self.get_scene_types(self.scene_id_to_type_path)

    def _init_callback(self) -> None:
        pass

    @staticmethod
    def get_scene_types(scene_id_to_type_path: str) -> List[List[str]]:
        """Construct a list mapping scene types to their corresponding types.

        :param scene_id_to_type_path: path to the mapping.
        :return: list of scene type tags per scene
        """
        # Read csv
        scene_type_dict: Dict[int, str]
        with open(scene_id_to_type_path, 'r') as f:
            csv_reader = csv.reader(f)
            scene_type_dict = {int(rows[0]): rows[1] for rows in csv_reader}

        # Convert dict to List[List[str]]
        scene_id_to_type_list: List[List[str]] = []
        for k, v in scene_type_dict.items():
            scene_id_to_type_list.append([v])
        return scene_id_to_type_list

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate episode outputs
            self.evaluate_scenes()

            # Start logging metrics, validators and composite metrics
            assert self.logger is not None

            # Aggregate metrics (ADE, FDE)
            ade_error, fde_error = self.compute_ade_fde(self.metric_set)
            self.logger.record(f'{self.prefix}/ade', ade_error)
            self.logger.record(f'{self.prefix}/fde', fde_error)

            # Aggregate validators
            validation_results = self.metric_set.evaluator.validation_results()
            agg = ValidationCountingAggregator().aggregate(validation_results)
            for k, v in agg.items():
                self.logger.record(f'{self.prefix}/{k}', v.item())
            # Add total collisions as well
            tot_collision = agg['collision_front'].item() + agg['collision_side'].item() + agg['collision_rear'].item()
            self.logger.record(f'{self.prefix}/total_collision', tot_collision)

            # Aggregate composite metrics
            composite_metric_results = self.metric_set.evaluator.composite_metric_results()
            comp_agg = CompositeMetricAggregator().aggregate(composite_metric_results)
            for k, v in comp_agg.items():
                self.logger.record(f'{self.prefix}/{k}', v.item())

            # If we should compute the scene-type aggregation metrics
            if self.enable_scene_type_aggregation:
                scene_type_results = \
                    compute_cle_scene_type_aggregations(self.metric_set,
                                                        self.scene_ids_to_scene_types,
                                                        list_validator_table_to_publish=[])
                for k, v in scene_type_results.items():
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
                    if info[idx]["sim_outs"][0].scene_id not in \
                            self.metric_set.evaluator.scene_metric_results.keys():
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

    @staticmethod
    def compute_ade_fde(metric_set: L5MetricSet) -> Tuple[float, float]:
        """Aggregate the Average displacement error (ADE) and Final displacement error (FDE)
        of the simulation outputs.

        :return: Tuple [ADE, FDE]"""
        scenes_result = metric_set.evaluator.metric_results()
        scene_ade_list: List[float] = []
        scene_fde_list: List[float] = []
        for _, scene_result in scenes_result.items():
            scene_ade_list.append(scene_result["displacement_error_l2"][1:].mean().item())
            scene_fde_list.append(scene_result['displacement_error_l2'][-1].item())

        if len(scene_ade_list) == 0:
            return (0, 0)

        average_ade = sum(scene_ade_list) / len(scene_ade_list)
        average_fde = sum(scene_fde_list) / len(scene_fde_list)

        return (average_ade, average_fde)
