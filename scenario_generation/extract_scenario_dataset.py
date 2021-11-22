import os

import numpy as np
from pathlib import Path
import torch
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.data import get_dataset_path
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.visualization.visualizer.zarr_utils import zarr_to_visualizer_scene
from bokeh import plotting

source_dataset_name = "val_data_loader"
sample_config = "/examples/urban_driver/config.yaml"

########################################################################
# Load data and configurations
########################################################################
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"], project_dir = get_dataset_path()
dm = LocalDataManager(None)
cfg = load_config_data(project_dir + sample_config)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_cfg = cfg[source_dataset_name]
dataset_zarr = ChunkedDataset(dm.require(dataset_cfg["key"])).open()
n_scenes = len(dataset_zarr.scenes)



vectorizer = build_vectorizer(cfg, dm)
dataset = EgoDatasetVectorized(cfg, dataset_zarr, vectorizer)

print(dataset)
print(f'Dataset source: {cfg[source_dataset_name]["key"]}, number of scenes total: {n_scenes}')

num_simulation_steps = 1
sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                           distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                           start_frame_index=0, show_info=True)

####################################################################################
# Data format documentation: https://github.com/ramitnv/l5kit/blob/master/docs/data_format.rst
####################################################################################

####################################################################################
num_scenes_limit = 20  # for debug

scene_idx = 13
# for scene_idx...........

# we inspect one scene at aa time (otherwise the program run may stuck)
scene_indices = [scene_idx]

sim_dataset = SimulationDataset.from_dataset_indices(dataset, scene_indices, sim_cfg)
frame_index = 0  # we need only t==0

ego_input = sim_dataset.rasterise_frame_batch(frame_index)[0]
agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)

other_agents_ids = [i for i in ego_input['host_id'] if i != 0]

n_other_agents = len(ego_input['host_id'])
print('n agents the ego sees = ', n_other_agents)
i_agent = 1
agents_input_lst = []
agents_ids_scene = [ky[1] for ky in agents_input.keys()]
agents_track_ids = []
print('agents_ids_scene = ', agents_ids_scene)
for i_agent in agents_ids_scene:
    cur_agent_input = agents_input[(scene_idx, i_agent)]
    agents_input_lst.append(cur_agent_input)
    agents_track_ids.append(cur_agent_input['track_id'])
print('n agent in scene = ', len(agents_input))

pass



# agents_feat = dict()
# # The coordinates (in agent reference system) of the AV in the future. Unit is meters
# agents_feat['positions'] = ego_input['target_positions']
#
# map_feat = dict()
# map_image = ego_input['image']
# # TODO: extract binary map?
#
# scene_save_dict = {'zarr_data': scene_dataset, 'agents_feat': agents_feat, 'map_feat': map_feat}




####################################################################################
# plot
####################################################################################
figure_path = Path('custom_filename' + '.html')
plotting.output_file(figure_path, title="Static HTML file")
fig = plotting.figure(sizing_mode="stretch_width", max_width=500, height=250)
output_notebook()
mapAPI = MapAPI.from_cfg(dm, cfg)

scene_dataset = dataset_zarr.get_scene_dataset(scene_idx)
vis_in = zarr_to_visualizer_scene(scene_dataset, mapAPI, with_trajectories=True)
vis_out = visualize(scene_idx, vis_in)
layout, fig = vis_out[0], vis_out[1]
show(layout)
plotting.save(fig)
print('Figure saved at ', figure_path)