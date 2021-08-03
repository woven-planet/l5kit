import os
import pickle
from typing import List, Optional

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from l5kit.environment.envs.l5_env import L5Env, SimulationOutputGym


def get_callback_list(output_prefix: str, n_envs: int, save_freq: int = 50000,
                      ckpt_prefix: Optional[str] = None) -> CallbackList:
    """ Generate the callback list to be used during model training in L5Kit gym.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param output_prefix: the prefix to save the model outputs during training
    :param n_envs: the number of parallel environments being used
    :param save_freq: the frequency to save the model and the outputs
    :param ckpt_prefix: the prefix to save the model during training
    """
    callback_list: List[BaseCallback] = []
    # Define callbacks
    assert output_prefix is not None, "Provide output prefix to save model states"

    # Save SimulationOutputGym periodically
    viz_callback = VizCallback(save_freq=(save_freq // n_envs), save_path='./logs/',
                               name_prefix=output_prefix, verbose=2)
    callback_list.append(viz_callback)

    # Save Model Periodically
    ckpt_prefix = ckpt_prefix if ckpt_prefix is not None else output_prefix
    checkpoint_callback = CheckpointCallback(save_freq=(save_freq // n_envs), save_path='./logs/',
                                             name_prefix=ckpt_prefix, verbose=2)
    callback_list.append(checkpoint_callback)

    # Save Model Config
    log_callback = LoggingCallback(output_prefix)
    callback_list.append(log_callback)

    callback = CallbackList(callback_list)
    return callback


class VizCallback(BaseCallback):
    """
    Callback for saving SimulationOutputGym every ``save_freq`` calls to ``env.step()``.
    The SimulationOutputGym will then be used for visualization.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the viz will be saved.
    :param name_prefix: Common prefix to the saved viz
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model_viz", verbose: int = 0) -> None:
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

    def _rollout_scene(self, idx: int) -> SimulationOutputGym:
        """ Rollout a particular scene index and return the simulation output.

        :param idx: the scene index to be rolled out
        :return: the simulation output of the rolled out scene
        """

        # Assert
        assert self.model is not None, "Model should be provided to VizCallback"
        assert isinstance(self.model.eval_env, L5Env), "Eval environment should be an instance of L5Env"
        assert 'reset_scene_id' in self.model.eval_env.__dict__.keys(), "Missing attribute 'reset_scene_id'"

        # Set the reset_scene_id to 'idx'
        self.model.eval_env.reset_scene_id = idx
        obs = self.model.eval_env.reset()
        for i in range(350):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.model.eval_env.step(action)
            if done:
                break

        sim_out: SimulationOutputGym
        sim_out = info["sim_outs"][0]
        return sim_out

    def _determine_rollout_scenes(self) -> List[int]:
        """ Determine the list of scene indices to be rolled out based on environment configuration.

        :return: the list of scene indices to be rolled out
        """

        # Assert
        assert self.model is not None, "Model should be provided to VizCallback"
        assert isinstance(self.model.eval_env, L5Env), "Eval environment should be an instance of L5Env"
        assert 'overfit' in self.model.eval_env.__dict__.keys(), "Missing attribute 'overfit'"
        assert 'max_scene_id' in self.model.eval_env.__dict__.keys(), "Missing attribute 'max_scene_id'"

        if self.model.eval_env.overfit:
            assert 'overfit_scene_id' in self.model.eval_env.__dict__.keys(), "Missing attribute 'overfit_scene_id'"
            return [self.model.eval_env.overfit_scene_id]

        scene_id_list = list(range(self.model.eval_env.max_scene_id))
        return scene_id_list


class LoggingCallback(BaseCallback):
    """
    Callback for mapping the tensorboard logger to model output filename at start of training.
    A callback is required as the tensorboard logger file is created just before training starts.

    :param output_prefix: the prefix for saving the current model being trained
    :param log_file: the file that contains the mapping between tensorboard logs and model outputs
    :param verbose:
    """

    def __init__(self, output_prefix: str, log_file: str = 'model_runs.txt', verbose: int = 0) -> None:
        super(LoggingCallback, self).__init__(verbose)
        self.output_prefix = output_prefix
        self.log_file = log_file

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        pass

    def _on_training_start(self) -> None:
        # Assert
        assert self.model is not None
        assert self.model.logger is not None

        with open(self.log_file, 'a') as f:
            f.write('{} \t \t \t {} \n \n'.format(self.model.logger.dir, self.output_prefix))
