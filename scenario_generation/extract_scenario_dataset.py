from pathlib import Path
import numpy as np
import torch
from bokeh import plotting
from bokeh.io import output_notebook, show
import matplotlib.pyplot as plt
from l5kit.data import MapAPI
from l5kit.geometry.transform import transform_point
from l5kit.simulation.dataset import SimulationDataset
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.visualization.visualizer.zarr_utils import zarr_to_visualizer_scene

from visualization_utils import visualize_scene_feat
####################################################################################

def mat_to_list_of_tensors(mat, mat_valid, coord_dim=2):
    list_tnsr = []
    for i in range(mat.shape[0]):
        lst = []
        for j in range(mat.shape[1]):
            if not mat_valid[i, j]:
                continue
            lst.append(mat[i, j])
        if not lst:
            continue
        lst = [torch.Tensor(elem[:coord_dim]) for elem in lst]
        tnsr = torch.stack(lst)
        list_tnsr.append(tnsr)

    return list_tnsr


####################################################################################
def get_scenes_batch(scene_indices_all, dataset, dataset_zarr, dm, sim_cfg, cfg, verbose=0, show_html_plot=False):
    """
    Data format documentation: https://github.com/ramitnv/l5kit/blob/master/docs/data_format.rst
    """


    data_generation_params = cfg['data_generation_params']
    other_agents_num = data_generation_params['other_agents_num']
    lane_params = data_generation_params['lane_params']
    max_num_crosswalks = lane_params['max_num_crosswalks']
    max_num_elem = max(lane_params['max_num_lanes'], max_num_crosswalks)  # max num elements per poly type
    max_points_per_crosswalk = lane_params['max_points_per_crosswalk']
    max_points_per_lane = lane_params['max_points_per_lane']
    max_points_per_elem = max(max_points_per_lane, max_points_per_crosswalk)
    coord_dim = 2  # we will use only X-Y coordinates

    map_feat = []  # agents features per scene
    agents_feat = []  # map features per scene

    agent_types_labels = ['CAR', 'CYCLIST', 'PEDESTRIAN']
    type_id_to_label = {3: 'CAR', 12: 'CYCLIST',
                        14: 'PEDESTRIAN'}  # based on the labels ids in l5kit/build/lib/l5kit/data/labels.py

    labels_hist_pre_filter = np.zeros(17, dtype=int)
    labels_hist = np.zeros(3, dtype=int)

    for i_scene, scene_idx in enumerate(scene_indices_all):

        print(f'Extracting scene #{i_scene + 1} out of {len(scene_indices_all)}')
        sim_dataset = SimulationDataset.from_dataset_indices(dataset, [scene_idx], sim_cfg)
        frame_index = 2  # we need only the initial t, but to get the speed we need to start at frame_index = 2

        ego_input = sim_dataset.rasterise_frame_batch(frame_index)[0]
        agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)

        agents_input_lst = []
        agents_ids_scene = [ky[1] for ky in agents_input.keys()]

        if verbose:
            print('agents_ids_scene = ', agents_ids_scene)
            print('n agent in scene = ', len(agents_input))

        ego_from_world = ego_input['agent_from_world']
        ego_yaw = ego_input['yaw']  # yaw angle in the agent in ego coord system [rad]
        ego_speed = ego_input['speed']
        ego_extent = ego_input['extent']

        agents_feat.append([])
        # add the ego car (in ego coord system):
        agents_feat[-1].append({
            'agent_label_id': agent_types_labels.index('CAR'),  # The ego car has the same label "car"
            'yaw': 0.,  # yaw angle in the agent in ego coord system [rad]
            'centroid': np.array([0, 0]),  # x,y position of the agent in ego coord system [m]
            'speed': ego_speed,  # speed [m/s ?]
            'extent': ego_extent[:2]})  # [length, width]  [m]
        # loop over agents in current scene:
        for (scene_id, agent_id) in agents_input.keys():  # if there are other agents besides the ego
            assert scene_id == scene_idx
            cur_agent_in = agents_input[(scene_id, agent_id)]
            agents_input_lst.append(cur_agent_in)
            agent_type = int(cur_agent_in['type'])
            labels_hist_pre_filter[agent_type] += 1
            if agent_type in type_id_to_label.keys():
                agent_label = type_id_to_label[agent_type]
                agent_label_id = agent_types_labels.index(agent_label)
            else:
                continue  # skip other agents types
            centroid_in_world = cur_agent_in['centroid']
            centroid = transform_point(centroid_in_world, ego_from_world)  # translation and rotation to ego system
            yaw_in_world = cur_agent_in['yaw']  # translation and rotation to ego system
            yaw = yaw_in_world - ego_yaw
            speed = cur_agent_in['speed']
            extent = cur_agent_in['extent']
            agents_feat[-1].append({
                'agent_label_id': agent_label_id,  # index of label in agent_types_labels
                'yaw': yaw,  # yaw angle in the agent in ego coord system [rad]
                'centroid': centroid,  # x,y position of the agent in ego coord system [m]
                'speed': speed,  # speed [m/s ?]
                'extent': extent[:2]  # [length, width]  [m]
            })
        # Get map features:
        lanes_mid = mat_to_list_of_tensors(ego_input['lanes_mid'], ego_input['lanes_mid_availabilities'])
        lanes_left = mat_to_list_of_tensors(ego_input['lanes'][::2], ego_input['lanes_availabilities'][::2])
        lanes_right = mat_to_list_of_tensors(ego_input['lanes'][1::2], ego_input['lanes_availabilities'][1::2])
        crosswalks = mat_to_list_of_tensors(ego_input['crosswalks'], ego_input['crosswalks_availabilities'])

        map_feat.append({'lanes_mid': lanes_mid,
                         'lanes_left': lanes_left,
                         'lanes_right': lanes_right,
                         'crosswalks': crosswalks,
                         })

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4

        polygon_types = ['lanes_mid', 'lanes_left', 'lanes_right', 'crosswalks']
        n_polygon_types = len(polygon_types)

        map_points = torch.zeros(n_polygon_types, max_num_elem, max_points_per_elem, coord_dim)
        map_points_availability = torch.zeros(n_polygon_types, max_num_elem, dtype=torch.bool)

        map_points[polygon_types.index('lanes_mid')] = torch.Tensor(ego_input['lanes_mid'][:, :, :coord_dim])
        map_points[polygon_types.index('lanes_left')] = torch.Tensor(ego_input['lanes'][::2, :, :coord_dim])
        map_points[polygon_types.index('lanes_right')] = torch.Tensor(ego_input['lanes'][1::2, :, :coord_dim])
        map_points[polygon_types.index('crosswalks')][:max_num_crosswalks, max_points_per_crosswalk] = \
            torch.Tensor(ego_input['crosswalks'][:, :, :coord_dim])
        # TODO: save map_points_availability
        # TODO: circular wrap to max_num_points for closed polygons?
        # map_points_availability[]


        # for i_poly_type, poly_type in enumerate(polygon_types):
        #     map_feat[i_poly_type] =

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



        if verbose and i_scene == 0:
            if show_html_plot:
                visualize_scene(dataset_zarr, cfg, dm, scene_idx)
            visualize_scene_feat(agents_feat[-1], map_feat[-1])

    print('labels_hist before filtering: ', {i: c for i, c in enumerate(labels_hist_pre_filter) if c > 0})
    print('labels_hist: ', {i: c for i, c in enumerate(labels_hist) if c > 0})
    return agents_feat, map_feat, agent_types_labels, labels_hist


####################################################################################

def visualize_scene(dataset_zarr, cfg, dm, scene_idx):
    figure_path = Path('loaded_scene' + '.html')
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


# Debug
# plt.subplot(311)
# plt.imshow(ego_input['lanes_availabilities'])
# plt.subplot(312)
# plt.imshow(ego_input['lanes'][:, :, 0] != 0.0)
# plt.subplot(313)
# plt.imshow(ego_input['lanes'][:, :, 1] != 0.0)
# # plt.show()
#
#
