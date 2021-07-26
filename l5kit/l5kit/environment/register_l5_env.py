import gym

from l5kit.environment import reward
from l5kit.simulation.dataset import SimulationConfig


def create_l5_env(env_config_path: str, eps_length: int, disable_cle: bool, rew_clip: float) -> None:
    """ Create and Register the L5Kit Gym-compatible environment """

    # Closed loop environment
    close_loop_envt = not disable_cle

    # Define Reward Function
    reward_fn: reward.Reward
    reward_fn = reward.CLE_Reward(rew_clip_thresh=rew_clip)


    # Define Close-Loop Simulator
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                               distance_th_far=30, distance_th_close=15,
                               num_simulation_steps=eps_length,
                               start_frame_index=0, show_info=True)

    # Register L5 Env
    gym.envs.register(
        id='L5-CLE-v0',
        entry_point="l5kit.environment.envs.l5_env:L5Env",
        kwargs={'env_config_path': env_config_path,
                'sim_cfg': sim_cfg,
                'reward': reward_fn,
                'cle': close_loop_envt},
    )
