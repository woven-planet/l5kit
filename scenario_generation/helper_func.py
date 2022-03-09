from pathlib import Path
import numpy as np
from bokeh import plotting
from bokeh.io import output_notebook, show
import l5kit.data as l5kit_data
import l5kit.visualization.visualizer.visualizer as visualizer
import l5kit.visualization.visualizer.zarr_utils as zarr_utils

####################################################################################


# def is_agent_on_road():
#
#     i_elem, i_point, mid_pos = find_closest_mid_lane()
#     right_lane_pos =
#     left_lane_pos =
#     tol = 1.1
#     allowed_deviate =  dist(lef_lane_pos, right_lane_pos) * 0.5 * tol + agent_length
#     agent_deviate = dist(mid_pos, agent_pos)
#     return  agent_deviate <= allowed_deviate


####################################################################################
def find_closest_mid_lane():
    return True





####################################################################################
def is_valid_agent(speed, extent, min_extent_length, min_extent_width):
    if speed < 0:
        return False
    length, width = extent
    return length >= min_extent_length and width >= min_extent_width and find_closest_mid_lane()
    # if length >= min_extent_length and width >= min_extent_width:
    #     print(f'Good: {length} x {width}')
    #     return True
    # else:
    #     print(f'Bad: {length} x {width}')
    #     return False



####################################################################################
def agent_feat_dict_to_vec(agent_dict, agent_feat_vec_coord_labels):
    dim_agent_feat_vec = len(agent_feat_vec_coord_labels)
    assert agent_feat_vec_coord_labels == ['centroid_x', 'centroid_y', 'yaw_cos', 'yaw_sin',
                                           'extent_length', 'extent_width', 'speed',
                                           'is_CAR', 'is_CYCLIST', 'is_PEDESTRIAN']
    agent_feat_vec = np.zeros(dim_agent_feat_vec, dtype=np.float32)
    agent_feat_vec[0] = agent_dict['centroid'][0]
    agent_feat_vec[1] = agent_dict['centroid'][1]
    agent_feat_vec[2] = np.cos(agent_dict['yaw'])
    agent_feat_vec[3] = np.sin(agent_dict['yaw'])
    agent_feat_vec[4] = float(agent_dict['extent'][0])
    agent_feat_vec[5] = float(agent_dict['extent'][1])
    agent_feat_vec[6] = float(agent_dict['speed'])
    # agent type ['CAR', 'CYCLIST', 'PEDESTRIAN'] is represented in one-hot encoding
    agent_feat_vec[7] = agent_dict['agent_label_id'] == 0
    agent_feat_vec[8] = agent_dict['agent_label_id'] == 1
    agent_feat_vec[9] = agent_dict['agent_label_id'] == 2
    assert agent_feat_vec[7:].sum() == 1

    return agent_feat_vec


####################################################################################

def visualize_scene(dataset_zarr, cfg, dm, scene_idx):
    figure_path = Path('loaded_scene' + '.html')
    plotting.output_file(figure_path, title="Static HTML file")
    fig = plotting.figure(sizing_mode="stretch_width", max_width=500, height=250)
    output_notebook()
    mapAPI = l5kit_data.MapAPI.from_cfg(dm, cfg)

    scene_dataset = dataset_zarr.get_scene_dataset(scene_idx)
    vis_in = zarr_utils.zarr_to_visualizer_scene(scene_dataset, mapAPI, with_trajectories=True)
    vis_out = visualizer.visualize(scene_idx, vis_in)
    layout, fig = vis_out[0], vis_out[1]
    show(layout)
    plotting.save(fig)
    print('Figure saved at ', figure_path)


####################################################################################


def get_poly_elems(ego_input, poly_type, dataset_props):
    max_num_elem = dataset_props['max_num_elem']
    max_points_per_elem = dataset_props['max_points_per_elem']
    coord_dim = dataset_props['coord_dim']

    elems_points = np.zeros((max_num_elem, max_points_per_elem, coord_dim), dtype=np.float32)
    is_points_valid = np.zeros((max_num_elem, max_points_per_elem), dtype=np.bool_)

    if poly_type == 'lanes_left':
        points = ego_input['lanes'][::2, :, :coord_dim]
        points_valid = ego_input['lanes_availabilities'][::2, :]
    elif poly_type == 'lanes_right':
        points = ego_input['lanes'][1::2, :, :coord_dim]
        points_valid = ego_input['lanes_availabilities'][1::2, :]
    else:
        points = ego_input[poly_type][:, :, :coord_dim]
        points_valid = ego_input[poly_type + '_availabilities'][:, :]

    elems_points[:points.shape[0], :points.shape[1], :] = points
    is_points_valid[:points_valid.shape[0], :points_valid.shape[1]] = points_valid
    return elems_points, is_points_valid

####################################################################################

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
