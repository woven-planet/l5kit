import numpy as np

import l5kit.geometry.transform as geometry_transform
import l5kit.simulation.dataset as simulation_dataset

from helper_func import visualize_scene, get_poly_elems, agent_feat_dict_to_vec, is_agent_valid
from visualization_utils import visualize_scene_feat


####################################################################################
def process_scenes_data(scene_indices_all, dataset, dataset_zarr, dm, sim_cfg, cfg, min_n_agents, max_n_agents,
                        min_extent_length,  min_extent_width, verbose=0):
    """
    Data format documentation: https://github.com/ramitnv/l5kit/blob/master/docs/data_format.rst
    """
    n_scenes_orig = len(scene_indices_all)
    data_generation_params = cfg['data_generation_params']
    lane_params = data_generation_params['lane_params']
    max_num_crosswalks = lane_params['max_num_crosswalks']
    max_num_lanes = lane_params['max_num_lanes']
    max_num_elem = max(max_num_lanes, max_num_crosswalks)  # max num elements per poly type
    max_points_per_crosswalk = lane_params['max_points_per_crosswalk']
    max_points_per_lane = lane_params['max_points_per_lane']
    max_points_per_elem = 2 * max(max_points_per_lane, max_points_per_crosswalk)  # we multiply by two, so the seq
    # will include its reflection fo the original seq as well
    coord_dim = 2  # we will use only X-Y coordinates
    agent_feat_vec_coord_labels = ['centroid_x',  # [0]  Real number
                                   'centroid_y',  # [1]  Real number
                                   'yaw_cos',  # [2]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
                                   'yaw_sin',  # [3]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
                                   'extent_length',  # [4] Real positive
                                   'extent_width',  # [5] Real positive
                                   'speed',  # [6] Real non-negative
                                   'is_CAR',  # [7] 0 or 1
                                   'is_CYCLIST',  # [8] 0 or 1
                                   'is_PEDESTRIAN',  # [9]  0 or 1
                                   ]
    dim_agent_feat_vec = len(agent_feat_vec_coord_labels)
    polygon_types = ['lanes_mid', 'lanes_left', 'lanes_right', 'crosswalks']
    closed_polygon_types = ['crosswalks']
    n_polygon_types = len(polygon_types)
    agent_types_labels = ['CAR', 'CYCLIST', 'PEDESTRIAN']
    type_id_to_label = {3: 'CAR', 12: 'CYCLIST',
                        14: 'PEDESTRIAN'}  # based on the labels ids in l5kit/build/lib/l5kit/data/labels.py
    dataset_props = {
        'polygon_types': polygon_types,
        'closed_polygon_types': closed_polygon_types,
        'max_num_elem': max_num_elem,
        'max_points_per_elem': max_points_per_elem,
        'agent_feat_vec_coord_labels': agent_feat_vec_coord_labels,
        'agent_types_labels': agent_types_labels,
        'coord_dim': coord_dim,
        'max_n_agents': max_n_agents,
        'dim_agent_feat_vec': dim_agent_feat_vec,
        'min_extent_length': min_extent_length,
        'min_extent_width': min_extent_width,
    }

    map_elems_points = np.zeros((n_scenes_orig, n_polygon_types, max_num_elem, max_points_per_elem, coord_dim),
                                dtype=np.float32)
    map_elems_n_points_orig = np.zeros((n_scenes_orig, n_polygon_types, max_num_elem), dtype=np.int16)
    map_elems_exists = np.zeros((n_scenes_orig, n_polygon_types, max_num_elem), dtype=np.bool_)
    agents_feat_vecs = np.zeros((n_scenes_orig, max_n_agents, dim_agent_feat_vec), dtype=np.float32)
    agents_num = np.zeros(n_scenes_orig, dtype=np.int16)
    agents_exists = np.zeros((n_scenes_orig, max_n_agents), dtype=np.bool_)

    labels_hist_pre_filter = np.zeros(17, dtype=int)
    labels_hist = np.zeros(3, dtype=int)
    ind_scene = 0  # number of valid scenes seen so far

    for i_scene, scene_idx in enumerate(scene_indices_all):

        # ------ debug display -----------#
        if verbose and i_scene == 2:
            visualize_scene(dataset_zarr, cfg, dm, scene_idx)

        print(f'Extracting scene #{i_scene + 1} out of {len(scene_indices_all)}')
        sim_dataset = simulation_dataset.SimulationDataset.from_dataset_indices(dataset, [scene_idx], sim_cfg)
        frame_index = 2  # we need only the initial t, but to get the speed we need to start at frame_index = 2

        ego_input = sim_dataset.rasterise_frame_batch(frame_index)[0]
        agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)

        # ---------  Get map data --------------#
        for i_type, poly_type in enumerate(polygon_types):
            elems_points, elems_points_valid = get_poly_elems(ego_input, poly_type, dataset_props)
            ind_elem = 0
            for i_elem in range(max_num_elem):
                elem_points = elems_points[i_elem]
                elem_points_valid = elems_points_valid[i_elem]
                n_valid_points = elem_points_valid.sum()
                if n_valid_points == 0:
                    continue
                map_elems_exists[ind_scene, i_type, ind_elem] = np.True_
                map_elems_n_points_orig[ind_scene, i_type, ind_elem] = n_valid_points
                # concatenate a reflection of this sequence, to create a shift equivariant representation
                # Note: since the new seq include the original + reflection, then we get flip invariant pipeline if we use
                # later cyclic shift invariant model
                # we keep adding flipped sequences to fill all points
                point_seq = elem_points[elem_points_valid]
                point_seq_flipped = point_seq[1:-2:-1]  # no need to duplicate edge points to get circular seq
                ind_point = 0
                is_flip = False
                while ind_point < max_points_per_elem:
                    if is_flip:
                        point_seq_cur = point_seq_flipped
                        seq_len = min(point_seq_flipped.shape[0], max_points_per_elem - ind_point)
                    else:
                        point_seq_cur = point_seq
                        seq_len = min(point_seq.shape[0], max_points_per_elem - ind_point)
                    map_elems_points[ind_scene, i_type, ind_elem, ind_point:(ind_point + seq_len)] \
                        = point_seq_cur[:seq_len]
                    ind_point += seq_len
                    is_flip = not is_flip
                ind_elem += 1

        # ---------  Get agents data --------------#
        agents_input_lst = []
        ego_from_world = ego_input['agent_from_world']
        ego_yaw = ego_input['yaw']  # yaw angle in the agent in ego coord system [rad]
        ego_speed = ego_input['speed']
        ego_extent = ego_input['extent'][:2]
        ego_centroid = np.array([0, 0])
        agents_feat_dicts = []  # the agents in this scene
        # add the ego car (in ego coord system):
        if is_agent_valid(ego_centroid, ego_speed, ego_extent, dataset_props,
                          map_elems_exists, map_elems_points, ind_scene):
            agents_feat_dicts.append({
                'agent_label_id': agent_types_labels.index('CAR'),  # The ego car has the same label "car"
                'yaw': 0.,  # yaw angle in the agent in ego coord system [rad]
                'centroid': ego_centroid,  # x,y position of the agent in ego coord system [m]
                'speed': ego_speed,  # speed [m/s ?]
                'extent': ego_extent})  # [length, width]  [m]
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
            centroid = geometry_transform.transform_point(centroid_in_world,
                                                          ego_from_world)  # translation and rotation to ego system
            yaw_in_world = cur_agent_in['yaw']  # translation and rotation to ego system
            yaw = yaw_in_world - ego_yaw
            speed = cur_agent_in['speed']
            extent = cur_agent_in['extent'][:2]
            if is_agent_valid(centroid, speed, extent, dataset_props,
                              map_elems_exists, map_elems_points, ind_scene):
                agents_feat_dicts.append({
                    'agent_label_id': agent_label_id,  # index of label in agent_types_labels
                    'yaw': yaw,  # yaw angle in the agent in ego coord system [rad]
                    'centroid': centroid,  # x,y position of the agent in ego coord system [m]
                    'speed': speed,  # speed [m/s ?]
                    'extent': extent  # [length, width]  [m]
                })
        n_valid_agents = len(agents_feat_dicts)
        if n_valid_agents < min_n_agents:
            continue  # discard this scene

        # Save the agents in order by the distance to ego
        agents_dists_to_ego = [np.linalg.norm(agent_dict['centroid'][:]) for agent_dict in agents_feat_dicts]
        agents_dists_order = np.argsort(agents_dists_to_ego)
        # we will use up to max_n_agents agents only from the data:
        agents_dists_order = agents_dists_order[:max_n_agents]
        for i_agent, i_agent_orig in enumerate(agents_dists_order):
            feat_vec = agent_feat_dict_to_vec(agents_feat_dicts[i_agent_orig], agent_feat_vec_coord_labels)
            agents_feat_vecs[ind_scene, i_agent] = feat_vec
            agents_exists[ind_scene, i_agent] = np.True_
        agents_num[ind_scene] = len(agents_dists_order)

        # ------ debug display -----------#
        if verbose and ind_scene == 8:
            print(f'n_valid_agents: {n_valid_agents}')
            visualize_scene_feat(agents_feat_dicts, map_elems_points[ind_scene], map_elems_exists[ind_scene],
                                 map_elems_n_points_orig[ind_scene], dataset_props)
        ind_scene += 1

    n_scenes = ind_scene
    saved_mats = {'map_elems_points': map_elems_points[:n_scenes],
                  'map_elems_n_points_orig': map_elems_n_points_orig[:n_scenes],
                  'map_elems_exists': map_elems_exists[:n_scenes],
                  'agents_feat_vecs': agents_feat_vecs[:n_scenes],
                  'agents_num': agents_num[:n_scenes],
                  'agents_exists': agents_exists[:n_scenes]}

    dataset_props['n_scenes'] = n_scenes
    print('labels_hist before filtering: ', {i: c for i, c in enumerate(labels_hist_pre_filter) if c > 0})
    print('labels_hist: ', {i: c for i, c in enumerate(labels_hist) if c > 0})
    return saved_mats, dataset_props, labels_hist

####################################################################################
