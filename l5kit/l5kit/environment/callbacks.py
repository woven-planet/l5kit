from typing import List, Optional

import gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback


def get_callback_list(eval_env: gym.Env, output_prefix: str, n_envs: int, save_freq: int = 50000,
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

    # Save Model Periodically
    ckpt_prefix = ckpt_prefix if ckpt_prefix is not None else output_prefix
    checkpoint_callback = CheckpointCallback(save_freq=(save_freq // n_envs), save_path='./logs/',
                                             name_prefix=ckpt_prefix, verbose=2)
    callback_list.append(checkpoint_callback)

    # Save Model Config
    log_callback = LoggingCallback(output_prefix)
    callback_list.append(log_callback)

    # Eval Model Config
    eval_callback = EvalCallback(eval_env, eval_freq=(save_freq // n_envs), n_eval_episodes=1000)
    callback_list.append(eval_callback)

    callback = CallbackList(callback_list)
    return callback


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
