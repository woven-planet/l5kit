import os
import subprocess
import pickle
import numpy as np
import l5kit.configs as l5kit_configs
import l5kit.data as l5kit_data
import general_util as general_util
import l5kit.dataset as l5kit_dataset
import l5kit.simulation.dataset as simulation_dataset
import l5kit.vectorization.vectorizer_builder as vectorizer_builder
from scenario_generation.extract_scenario_dataset import process_scenes_data

########################################################################
verbose = 0  # 0 | 1
show_html_plot = False
config_file_name = 'train'  # 'sample' | 'train' | 'train_full'
source_name = "train_data_loader"  # "train_data_loader | "val_data_loader"
save_dir_name = 'l5kit_data_' + config_file_name + '_' + source_name
sample_config = f"/scenario_generation/configs/config_{config_file_name}.yaml"

max_n_agents = 10  # we will use up to max_n_agents agents only from the data

# Our changes:
# max_retrieval_distance_m: 40  # maximum radius around the AoI for which we retrieve
# max_agents_distance: 40 # maximum distance from AoI for another agent to be picked
# train_data_loader key

########################################################################
save_folder = 'Saved_Data'
save_dir_path = os.path.join(save_folder, save_dir_name)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

if os.path.exists(save_dir_path):
    print(f'Save path {save_dir_path} already exists, overwriting...')
else:
    os.mkdir(save_dir_path)

map_data_file_path = os.path.join(save_dir_path, 'map_data.dat')
info_file_path = os.path.join(save_dir_path, 'info_data.pkl')

######### DEBUG ###########

# with open(info_file_path, 'rb') as fid:
#     info_dict = pickle.loads(fid.read())
# m_info = info_dict['saved_mats_info']['map_elems_points']
# # # Load the memmap data in Read-only mode:
# ewfp = np.memmap(m_info['path'], dtype=m_info['dtype'], mode='r', shape=m_info['shape'])


########################################################################
# Load data and configurations
########################################################################
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"], project_dir = general_util.get_dataset_path()
dm = l5kit_data.LocalDataManager(None)
cfg = l5kit_configs.load_config_data(project_dir + sample_config)

dataset_cfg = cfg[source_name]

dataset_zarr = l5kit_data.ChunkedDataset(dm.require(dataset_cfg["key"])).open()
n_scenes = len(dataset_zarr.scenes)
vectorizer = vectorizer_builder.build_vectorizer(cfg, dm)
dataset = l5kit_dataset.EgoDatasetVectorized(cfg, dataset_zarr, vectorizer)

print(dataset)
print(f'Dataset source: {cfg[source_name]["key"]}, number of scenes total: {n_scenes}')

num_simulation_steps = 10
sim_cfg = simulation_dataset.SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                                              distance_th_far=500, distance_th_close=50,
                                              num_simulation_steps=num_simulation_steps,
                                              start_frame_index=0, show_info=True)

scene_indices = list(range(n_scenes))
# scene_indices = [39]

saved_mats, dataset_props, labels_hist = process_scenes_data(scene_indices, dataset,
                                                             dataset_zarr,
                                                             dm, sim_cfg, cfg, max_n_agents,
                                                             verbose=verbose,
                                                             show_html_plot=show_html_plot)

git_version = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

saved_mats_info = {}
for var_name, var in saved_mats.items():
    save_file_path = os.path.join(save_dir_path, var_name, '.dat')
    # Create a memmap with dtype and shape that matches our data:
    fp = np.memmap(save_file_path, dtype=var.dtype, mode='w+', shape=var.shape)
    fp[:] = var[:]  # write data to memmap array
    fp.flush()  # Flushes memory changes to disk in order to read them back
    saved_mats_info[var_name] = {'path': save_file_path, 'dtype': var.dtype, 'shape': var.shape}

with open(info_file_path, 'wb') as fid:
    pickle.dump({'dataset_props': dataset_props, 'saved_mats_info': saved_mats_info,
                 'git_version': git_version, 'labels_hist': labels_hist}, fid)

print(f'Saved data of {len(scene_indices)} scenes at ', save_dir_path)
