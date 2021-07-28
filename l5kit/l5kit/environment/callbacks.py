import os
import pickle

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class VizCallback(BaseCallback):
    """
    Callback for saving SimulationOutputs of current model state every ``save_freq`` calls
    to ``env.step()``. The SimulationOutputs will then be used for visualization.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the viz will be saved.
    :param name_prefix: Common prefix to the saved viz
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model_viz", verbose: int = 0):
        super(VizCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            simulation_outputs = []
            scene_id_list = self._determine_rollout_scenes()

            for id_ in scene_id_list:
                sim_out = self._rollout_scene(id_)
                simulation_outputs.append(sim_out)

            if self.verbose > 1:
                print(f"Saving SimulationOutputGym to {path}")

            with open(path + ".pkl", 'wb') as f:
                pickle.dump(simulation_outputs, f)

        return True

    def _rollout_scene(self, idx: int):
        obs = self.model.eval_env.reset(scene_index=idx)
        for i in range(350):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.model.eval_env.step(action)
            if done:
                break

        sim_out = info["info"][0]
        return sim_out

    def _determine_rollout_scenes(self):
        if self.model.eval_env.overfit:
            return [self.model.eval_env.overfit_scene_id]

        scene_id_list = list(range(self.model.eval_env.max_scene_id))
        return scene_id_list


class TrajectoryCallback(BaseCallback):
    """
    Callback for saving trajectory at end of training. This trajectory will be used for L2 error calculation.
    Used only for OpenLoop training.

    :param save_freq:
    :param save_path: Path to the folder where the viz will be saved.
    :param name_prefix: Common prefix to the saved viz
    :param verbose:
    """

    def __init__(self, save_path: str, name_prefix: str = "rl_model_traj", verbose: int = 0):
        super(TrajectoryCallback, self).__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        pass

    def _on_training_end(self) -> bool:
        path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
        obs = self.model.eval_env.reset()
        action_list = []
        gt_action_list = []
        done = False
        for i in range(350):
            gt_action_list.append(self.model.eval_env.ego_input_dict["target_positions"][0, 0])

            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.model.eval_env.step(action)
            action_list.append(action[:2])
            if done:
                break

        action_list = np.stack(action_list)
        gt_action_list = np.stack(gt_action_list)

        error = np.square(action_list - gt_action_list).sum() / len(action_list)
        error = np.sqrt(error)
        print("Error: ", error)
        with open(path + ".pkl", 'wb') as f:
            pickle.dump([gt_action_list, action_list], f)

        return True


class LoggingCallback(BaseCallback):
    """
    Callback for logging model config at start of training.

    :param verbose:
    """

    def __init__(self, args, verbose: int = 0):
        super(LoggingCallback, self).__init__(verbose)
        self.args = args

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        pass

    def _on_training_start(self) -> bool:
        with open('model_runs.txt', 'a') as f:
            f.write(self.model.logger.dir)
            f.write('\t \t \t')
            f.write(self.args.output_prefix)
            f.write('\n \n')
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        env_rewards = self.model.env.get_attr('reward')
        for i, reward in enumerate(env_rewards):
            self.logger.record('reward/{}th_yaw_error'.format(i + 1), reward.yaw_error)
            self.logger.record('reward/{}th_dist_error'.format(i + 1), reward.dist_error)
        return True
