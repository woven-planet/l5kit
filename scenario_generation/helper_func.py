from pathlib import Path
import numpy as np
from bokeh import plotting
from bokeh.io import output_notebook, show
import l5kit.data as l5kit_data
import l5kit.visualization.visualizer.visualizer as visualizer
import l5kit.visualization.visualizer.zarr_utils as zarr_utils


####################################################################################

def is_agent_valid(centroid, speed, extent, dataset_props, map_elems_exists, map_elems_points, ind_scene, agent_name, verbose):
    if speed < 0:
        if verbose:
            print(f'Agent {agent_name} discarded - negative speed {speed}')
        return False
    min_extent_length = dataset_props['min_extent_length']
    min_extent_width = dataset_props['min_extent_width']
    length, width = extent
    if length < min_extent_length or width < min_extent_width:
        if verbose:
            print(f'Agent {agent_name} discarded, length: {length},'
                  f' width: {width}, min_extent_length: {min_extent_length}, min_extent_width: {min_extent_width}')
        return False
    polygon_types = dataset_props['polygon_types']
    i_lanes_mid = polygon_types.index('lanes_mid')
    i_lanes_left = polygon_types.index('lanes_left')
    i_lanes_right = polygon_types.index('lanes_right')

    lanes_mid_exists = map_elems_exists[ind_scene, i_lanes_mid]
    lanes_left_exists = map_elems_exists[ind_scene, i_lanes_left]
    lanes_right_exists = map_elems_exists[ind_scene, i_lanes_right]

    lanes_mid_points = map_elems_points[ind_scene, i_lanes_mid]
    lanes_mid_points = lanes_mid_points[lanes_mid_exists].reshape((-1, 2))
    lanes_left_points = map_elems_points[ind_scene, i_lanes_left]
    lanes_left_points = lanes_left_points[lanes_left_exists].reshape((-1, 2))
    lanes_right_points = map_elems_points[ind_scene, i_lanes_right]
    lanes_right_points = lanes_right_points[lanes_right_exists].reshape((-1, 2))

    # find the closest mid-lane point to the agent centroid
    dists_to_mid_points = np.linalg.norm(centroid - lanes_mid_points, axis=1)
    i_min_dist_to_mid = dists_to_mid_points.argmin()
    mid_point = lanes_mid_points[i_min_dist_to_mid]

    # disqualify the point if there is any left-lane or right-lane point closer to mid_point than the centroid
    min_dist_to_left = np.min(np.linalg.norm(mid_point - lanes_left_points, axis=1))
    min_dist_to_right = np.min(np.linalg.norm(mid_point - lanes_right_points, axis=1))

    dist_to_centroid = np.linalg.norm(mid_point - centroid)

    if dist_to_centroid > min_dist_to_left or dist_to_centroid > min_dist_to_right:
        if verbose:
            print(f'Agent {agent_name} discarded, dist_to_centroid: {dist_to_centroid},'
                  f' min_dist_to_left: {min_dist_to_left}, min_dist_to_right: {min_dist_to_right}')
        return False
    if verbose:
        print(f'Agent {agent_name} is OK')
    return True


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
    figure_path = Path(f'loaded_scene_idx_{scene_idx}' + '.html')
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
