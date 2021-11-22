import os
from pathlib import Path

import torch
from bokeh import plotting
from bokeh.io import output_notebook, show

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.data import MapAPI
from l5kit.data import get_dataset_path
from l5kit.dataset import EgoDatasetVectorized
from l5kit.geometry.transform import transform_point
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.visualization.visualizer.zarr_utils import zarr_to_visualizer_scene

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


agents_input_lst = []
agents_ids_scene = [ky[1] for ky in agents_input.keys()]

print('agents_ids_scene = ', agents_ids_scene)
print('n agent in scene = ', len(agents_input))

'''
agents_feat:
* track_id
* type
* centroid - # x,y position of the agent in ego coord system [m]
* yaw - # yaw angle   of the agent in ego coord system [rad]    
*  'extent' - [length?, width?]  [m]
'''

ego_from_world_rot_mat = ego_input['agent_from_world']
ego_centroid = ego_input['centroid']
ego_yaw = ego_input['yaw']

agents_feat = []
map_feat = []
for i_agent in agents_ids_scene:
    cur_agent_in = agents_input[(scene_idx, i_agent)]
    agents_input_lst.append(cur_agent_in)
    track_id = cur_agent_in['track_id']
    agent_type = cur_agent_in['type']
    centroid_in_world = cur_agent_in['centroid']
    centroid = transform_point(centroid_in_world, ego_from_world_rot_mat)  # translation and rotation to ego system
    yaw_in_world = cur_agent_in['yaw'] # translation and rotation to ego system
    yaw = yaw_in_world - ego_yaw
    speed = cur_agent_in['speed']
    extent = cur_agent_in['extent']
    agents_feat.append({'track_id': track_id, 'agent_type': agent_type, 'yaw': yaw,
                        'centroid': centroid, 'speed': speed, 'extent': extent})
    # TODO: transform to ego coords + add ego to list
    pass
print('agents centroids: ', [af['centroid'] for af in agents_feat])
print('agents yaws: ', [af['yaw'] for af in agents_feat])
# print('agents_feat: ', agents_feat)
# print('map_feat: ', map_feat)

# Get map features in ego coords
map_feat.append({'lanes': ego_input['lanes']})



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