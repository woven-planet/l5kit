import os
from pathlib import Path
import numpy as np
import torch
from bokeh import plotting
from bokeh.io import output_notebook, show
import matplotlib.pyplot as plt
import numpy as np

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

num_simulation_steps = 50
sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                           distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                           start_frame_index=0, show_info=True)

####################################################################################
# Data format documentation: https://github.com/ramitnv/l5kit/blob/master/docs/data_format.rst
####################################################################################

####################################################################################
num_scenes_limit = 100  # for debug

scene_idx = 34
# for scene_idx...........

# we inspect one scene at aa time (otherwise the program run may stuck)
scene_indices = [scene_idx]

sim_dataset = SimulationDataset.from_dataset_indices(dataset, scene_indices, sim_cfg)
frame_index = 2  # we need only the initial t, but to get the speed we need to start at frame_index = 2

ego_input = sim_dataset.rasterise_frame_batch(frame_index)[0]
agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)

agents_input_lst = []
agents_ids_scene = [ky[1] for ky in agents_input.keys()]

print('agents_ids_scene = ', agents_ids_scene)
print('n agent in scene = ', len(agents_input))

ego_from_world_rot_mat = ego_input['agent_from_world']
ego_centroid = ego_input['centroid']
ego_yaw = ego_input['yaw']
ego_speed = ego_input['speed']
ego_extent = ego_input['extent']

agents_feat = []
# add the ego car (in ego coord system):
agents_feat.append({'track_id': ego_input['track_id'], 'agent_type': ego_input['type'], 'yaw': 0.,
                    'centroid': np.array([0, 0]), 'speed': ego_input['speed'], 'extent': ego_input['extent']})

for i_agent in agents_ids_scene:
    cur_agent_in = agents_input[(scene_idx, i_agent)]
    agents_input_lst.append(cur_agent_in)
    track_id = cur_agent_in['track_id']
    agent_type = cur_agent_in['type']
    centroid_in_world = cur_agent_in['centroid']
    centroid = transform_point(centroid_in_world, ego_from_world_rot_mat)  # translation and rotation to ego system
    yaw_in_world = cur_agent_in['yaw']  # translation and rotation to ego system
    yaw = yaw_in_world - ego_yaw
    speed = cur_agent_in['speed']
    extent = cur_agent_in['extent']
    agents_feat.append({'track_id': track_id,
                        'agent_type': agent_type,
                        'yaw': yaw,  # yaw angle in the agent in ego coord system [rad]
                        'centroid': centroid,  # x,y position of the agent in ego coord system [m]
                        'speed': speed,  # speed [m/s ?]
                        'extent': extent})  # [length?, width?]  [m]

print('agents centroids: ', [af['centroid'] for af in agents_feat])
print('agents yaws: ', [af['yaw'] for af in agents_feat])
print('agents speed: ', [af['speed'] for af in agents_feat])
print('agents types: ', [af['agent_type'] for af in agents_feat])

###################################################################
# Debug
# plt.subplot(311)
# plt.imshow(ego_input['lanes_availabilities'])
# plt.subplot(312)
# plt.imshow(ego_input['lanes'][:, :, 0] != 0.0)
# plt.subplot(313)
# plt.imshow(ego_input['lanes'][:, :, 1] != 0.0)
# plt.show()
################################################################33
lane_x = []
lane_y = []
lane_left_x = []
lane_left_y = []
lane_right_x = []
lane_right_y = []
i_elem = 0
i_point = 0
for i_elem in range(ego_input['lanes'].shape[0]):
    lane_x.append([])
    lane_y.append([])
    for i_point in range(ego_input['lanes'].shape[1]):
        if not ego_input['lanes_availabilities'][i_elem, i_point]:
            continue
        lane_x[-1].append(ego_input['lanes'][i_elem, i_point, 0])
        lane_y[-1].append(ego_input['lanes'][i_elem, i_point, 1])
lane_x = [lst for lst in lane_x if lst != []]
lane_y = [lst for lst in lane_y if lst != []]

# transform_points(agents_input[(0, 1)]['lanes_mid'][:, :, :2], agents_input[(0, 1)]["world_from_agent"])[:2, :, :]transform_points(agents_input[(0, 1)]['lanes_mid'][:, :, :2], agents_input[(0, 1)]["world_from_agent"])[:2, :, :]

map_feat = {'lane_x': lane_x, 'lane_y': lane_y}

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

####################################################################################

import matplotlib.pyplot as plt

X = [af['centroid'][0] for af in agents_feat]
Y = [af['centroid'][1] for af in agents_feat]
U = [af['speed'] * np.cos(af['yaw']) for af in agents_feat]
V = [af['speed'] * np.sin(af['yaw']) for af in agents_feat]
fig, ax = plt.subplots()
ax.quiver(X, Y, U, V, units='xy', color='b')
ax.quiver(X[0], Y[0], U[0], V[0], units='xy', color='r')  # draw ego

for i_elem in range(len(map_feat['lane_x'])):
    x = map_feat['lane_x'][i_elem]
    y = map_feat['lane_y'][i_elem]
    ax.fill(x, y, facecolor='0.4', alpha=0.3, edgecolor='black', linewidth=1)

ax.grid()
plt.show()
pass
##############################################################################################3

# for i_elem in range(map_feat['n_lane_elem']):
#     left_x = map_feat['lane_left_x'][i_elem]
#     left_y = map_feat['lane_left_y'][i_elem]
#     right_x = map_feat['lane_right_x'][i_elem]
#     right_y = map_feat['lane_right_y'][i_elem]
#     x = np.concatenate((left_x, right_x))
#     y = np.concatenate((left_y, right_y))
#     ax.fill(x, y, facecolor='0.4', alpha=0.2)
# for i_elem in range(map_feat['n_lane_elem']):
#     x = map_feat['lane_x'][i_elem]
#     y = map_feat['lane_y'][i_elem]


# agents_feat = dict()
# # The coordinates (in agent reference system) of the AV in the future. Unit is meters
# agents_feat['positions'] = ego_input['target_positions']
#
# map_feat = dict()
# map_image = ego_input['image']
# # extract binary map?
#
# scene_save_dict = {'zarr_data': scene_dataset, 'agents_feat': agents_feat, 'map_feat': map_feat}


# Get map features in ego coords

# n_elements = ego_input['lanes'].shape[0]

# lane_x = ego_input['lanes'][0, :, 0]
# lane_y = ego_input['lanes'][0, :, 0]
#
# map_feat = {'lane_x': lane_x, 'lane_y': lane_y}

# # fields = ['lane_left_x', 'lane_left_y', 'lane_right_x', 'lane_right_y', 'lane_x', 'lane_y']
# fields = ['lane_x', 'lane_y']
#
# for fld in fields:
#     map_feat[fld] = []
# for i_elem in range(n_elements):
#     n_points = ego_input['lanes_availabilities'][i_elem, :].sum()
#     if n_points == 0:
#         continue
#     for fld in fields:
#         map_feat[fld].append([])
#     # for i_point in range(0, n_points, 2):
#     #     map_feat['lane_left_x'][i_elem].append(ego_input['lanes'][i_elem, i_point, 0])
#     #     map_feat['lane_left_y'][i_elem].append(ego_input['lanes'][i_point, i_elem, 1])
#     #     map_feat['lane_right_x'][i_elem].append(ego_input['lanes'][i_point + 1, i_elem, 0])
#     #     map_feat['lane_right_y'][i_elem].append(ego_input['lanes'][i_point + 1, i_elem, 1])
#     for i_point in range(n_points):
#         if ego_input['lanes_availabilities'][i_elem, i_point].sum():
#             map_feat['lane_x'][i_elem].append(ego_input['lanes'][i_elem, i_point, 0])
#             map_feat['lane_y'][i_elem].append(ego_input['lanes'][i_elem, i_point, 1])
#
# map_feat['n_lane_elem'] = len(map_feat['lane_left_x'])
