import time
import argparse
import os
import pickle

import gym
from stable_baselines3 import PPO

import l5kit.environment
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset, filter_agents_by_frames
from l5kit.dataset import EgoDataset
from l5kit.environment.envs.l5_env import SimulationConfigGym
from l5kit.environment.gym_metric_set import L2DisplacementYawMetricSet
from l5kit.environment.policy_utils import rollout_multiple_scenes
from l5kit.rasterization import build_rasterizer
from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle
from l5kit.visualization.visualizer.visualizer import visualize_gif
from l5kit.data import MapAPI

from bokeh.io import output_notebook, show
from prettytable import PrettyTable

from moviepy.editor import ImageSequenceClip
from IPython.display import display, Image

# model_prefix = 'verify_g080_s10_32_2400000_steps'
# model_prefix = 'verify_g080_s10_yaw5_32_2400000_steps'
model_prefix = 'verify_s10_lrsched_8_5000000_steps'
# model_prefix = 'verify_s10_8_4000000_steps'
# model_prefix = 'verify_s10_8_4000000_steps'
model_prefix = 'verify_s10_8_deepcnn_2000000_steps'
model_prefix = 'verify_yaw45_g080_deepcnn_s10_16_2800000_steps'
model_prefix = 'verify_yaw45_g080_deepcnn_s10_16_'
model_prefix = 'verify_g080_s10_yaw1_32_2800000_steps'
model_prefix = 'verify_yaw60_s10_g080_yaw1_32_5200000_steps'
model_prefix = 'verify_yaw45_s100_g080_yaw1_32_run3_75M_stepLR_3000000_steps'
model_prefix = 'verify_yaw60_s10_g080_yaw1_32_5200000_steps'

model_paths = []
PARENT_DIR = './logs/'
for filename in os.listdir(PARENT_DIR):
    if model_prefix in filename:
        model_paths.append(PARENT_DIR + filename.split(".")[0])
print(model_paths)

def gif_outputs(sim_outs, name='test'):
    sim_out = sim_outs[0] # for each scene
    print("Scene ID: ", sim_out.scene_id)
    vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, mapAPI)
    gif_frames = visualize_gif(sim_out.scene_id, vis_in)

    # save it as a gif
    clip = ImageSequenceClip(list(gif_frames), fps=50)
    clip.write_gif(name + '.gif', fps=50)


def quantify_outputs(sim_outs, metric_set=None):
    metric_set = metric_set if metric_set is not None else L2DisplacementYawMetricSet()

    metric_set.evaluate(sim_outs)
    scene_results = metric_set.evaluator.scene_metric_results
    fields = ["scene_id", "FDE", "ADE"]
    # table = PrettyTable(field_names=fields)
    tot_fde = 0.0
    tot_ade = 0.0
    for scene_id in scene_results:
        scene_metrics = scene_results[scene_id]
        ade_error = scene_metrics["displacement_error_l2"][1:].mean()
        fde_error = scene_metrics['displacement_error_l2'][-1]
    #     table.add_row([scene_id, round(fde_error.item(), 4), round(ade_error.item(), 4)])
    #     tot_fde += fde_error.item()
    #     tot_ade += ade_error.item()
        return ade_error.item(), fde_error.item()
    # ave_fde = tot_fde / len(scene_results)
    # ave_ade = tot_ade / len(scene_results)
    # table.add_row(["Overall", round(ave_fde, 4), round(ave_ade, 4)])
    # print(table)

# set env variable for data
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./gym_config.yaml")

mapAPI = MapAPI.from_cfg(dm, cfg)

env_config_path = './gym_config.yaml'

validation_sim_cfg = SimulationConfigGym()
eps_length = 240
# validation_sim_cfg.num_simulation_steps = eps_length + 1
validation_sim_cfg.num_simulation_steps = None
env = gym.make("L5-CLE-v0", env_config_path=env_config_path, use_kinematic=True, return_info=True, train=False,
               sim_cfg=validation_sim_cfg, randomize_start=False)


max_scene = 1
max_frame = 1
fde_thresh = 0.0

file_name = 'ade_fde.txt'

for model_path in model_paths:
    t_step = model_path.split('_')[-2]
    print("Time Step:", t_step)
    model = PPO.load(model_path, env)
    fde_list = []
    ade_list = []

    fields = ["scene_id", "eps_length", "ADE", "FDE"]
    table = PrettyTable(field_names=fields)

    # for s_idx in range(max_scene):
    for s_idx in [2, 3, 4, 9]:
        for f_idx in range(max_frame):
            # Set the reset_scene_id to 'idx'
            env.reset_scene_id = s_idx
            # Reset frame id
            env.sim_cfg.start_frame_index = f_idx

            obs = env.reset()
            for i in range(350):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                if done:
                    break

            sim_out = info["sim_outs"][0]
            import time
            st = time.time()
            gif_outputs([sim_out], name=str(s_idx))
            print(time.time() - st)

            ade_error, fde_error = quantify_outputs([sim_out])
            table.add_row([s_idx, eps_length, round(ade_error, 4), round(fde_error, 4)])
            fde_list.append(fde_error)
            ade_list.append(ade_error)
            # if fde_error >= fde_thresh:
                # print("ADE: ", ade_error, " FDE: ", fde_error)

    # print(len(fde_list), max(fde_list), sum(fde_list)/len(fde_list))
    table.add_row(["Overall", eps_length, round(sum(ade_list)/len(ade_list), 4), round(sum(fde_list)/len(fde_list), 4)])
    # print(model_path)
    # print(table)
    with open(file_name, 'a') as f:
        f.write(model_path)
        f.write("\n")
        f.write(str(table))
        f.write("\n\n\n")
