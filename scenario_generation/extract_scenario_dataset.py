
from pathlib import Path
import numpy as np
from bokeh import plotting
from bokeh.io import output_notebook, show
import matplotlib.pyplot as plt
from l5kit.data import MapAPI
from l5kit.geometry.transform import transform_point
from l5kit.simulation.dataset import  SimulationDataset
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.visualization.visualizer.zarr_utils import zarr_to_visualizer_scene


####################################################################################

def mat_to_list_of_lists(mat, mat_valid):
    lstlst = []
    for i in range(mat.shape[0]):
        lstlst.append([])
        for j in range(mat.shape[1]):
            if not mat_valid[i, j]:
                continue
            lstlst[-1].append(mat[i, j])
    lstlst = [lst for lst in lstlst if lst != []]
    return lstlst

####################################################################################
def get_scenes_batch(scene_indices_all, dataset, dataset_zarr, dm, sim_cfg, cfg, verbose=0):
    """
    Data format documentation: https://github.com/ramitnv/l5kit/blob/master/docs/data_format.rst
    """

    map_feat = []  # agents features per scene
    agents_feat = []  # map features per scene

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
        ego_yaw = ego_input['yaw']
        ego_speed = ego_input['speed']
        ego_extent = ego_input['extent']

        agents_feat.append([])
        # add the ego car (in ego coord system):
        agents_feat[-1].append({'track_id': ego_input['track_id'], 'agent_type': ego_input['type'], 'yaw': 0.,
                                'centroid': np.array([0, 0]), 'speed': ego_speed, 'extent': ego_extent})

        for (scene_id, agent_id) in agents_input.keys():  # if there are other agents besides the ego
            assert scene_id == scene_idx
            cur_agent_in = agents_input[(scene_id, agent_id)]
            agents_input_lst.append(cur_agent_in)
            track_id = cur_agent_in['track_id']
            agent_type = cur_agent_in['type']
            centroid_in_world = cur_agent_in['centroid']
            centroid = transform_point(centroid_in_world, ego_from_world)  # translation and rotation to ego system
            yaw_in_world = cur_agent_in['yaw']  # translation and rotation to ego system
            yaw = yaw_in_world - ego_yaw
            speed = cur_agent_in['speed']
            extent = cur_agent_in['extent']
            agents_feat[-1].append({'track_id': track_id,
                                    'agent_type': agent_type,
                                    'yaw': yaw,  # yaw angle in the agent in ego coord system [rad]
                                    'centroid': centroid,  # x,y position of the agent in ego coord system [m]
                                    'speed': speed,  # speed [m/s ?]
                                    'extent': extent})  # [length?, width?]  [m]
        # Get Lanes
        lanes_mid = mat_to_list_of_lists(ego_input['lanes_mid'], ego_input['lanes_mid_availabilities'])
        lanes_left = mat_to_list_of_lists(ego_input['lanes'][::2], ego_input['lanes_availabilities'][::2])
        lanes_right = mat_to_list_of_lists(ego_input['lanes'][1::2], ego_input['lanes_availabilities'][1::2])
        crosswalks = mat_to_list_of_lists(ego_input['crosswalks'], ego_input['crosswalks_availabilities'])

        map_feat.append({'lanes_mid': lanes_mid,
                         'lanes_left': lanes_left,
                         'lanes_right': lanes_right,
                         'crosswalks': crosswalks,
                         })

        if verbose and i_scene == 0:
            visualize_scene(dataset_zarr, cfg, dm, scene_idx)
            visualize_scene_feat(agents_feat[-1], map_feat[-1])

    return agents_feat, map_feat


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
####################################################################################


def plot_poly(ax, poly, facecolor='0.4', alpha=0.3, edgecolor='black'):
    for elem in poly:
        x = [p[0] for p in elem]
        y = [p[1] for p in elem]
        ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=1)


def visualize_scene_feat(agents_feat, map_feat):
    print('agents centroids: ', [af['centroid'] for af in agents_feat])
    print('agents yaws: ', [af['yaw'] for af in agents_feat])
    print('agents speed: ', [af['speed'] for af in agents_feat])
    print('agents types: ', [af['agent_type'] for af in agents_feat])
    X = [af['centroid'][0] for af in agents_feat]
    Y = [af['centroid'][1] for af in agents_feat]
    U = [af['speed'] * np.cos(af['yaw']) for af in agents_feat]
    V = [af['speed'] * np.sin(af['yaw']) for af in agents_feat]
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V, units='xy', color='b')
    ax.quiver(X[0], Y[0], U[0], V[0], units='xy', color='r')  # draw ego

    plot_poly(ax, map_feat['lanes_left'], facecolor='0.4', alpha=0.3, edgecolor='black')
    plot_poly(ax, map_feat['lanes_right'], facecolor='0.4', alpha=0.3, edgecolor='brown')
    plot_poly(ax, map_feat['lanes_mid'], facecolor='purple', alpha=0.3, edgecolor='purple')
    plot_poly(ax, map_feat['crosswalks'], facecolor='yellow', alpha=0.3, edgecolor='yellow')

    ax.grid()
    plt.show()
##############################################################################################3

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