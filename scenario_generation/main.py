import os
import subprocess
import pickle
import l5kit.configs as l5kit_configs
import l5kit.data as l5kit_data
import general_util as general_util
import l5kit.dataset as l5kit_dataset
import l5kit.simulation.dataset as simulation_dataset
import l5kit.vectorization.vectorizer_builder as vectorizer_builder
from scenario_generation.extract_scenario_dataset import process_scenes_data
from pathlib import Path
import h5py

########################################################################
verbose = 0  # 0 | 1
show_html_plot = False
config_file_name = 'sample'  # 'sample' | 'train' | 'train_full'
source_name = "train_data_loader"  # "train_data_loader | "val_data_loader"
save_dir_name = 'l5kit_data_' + config_file_name + '_' + source_name
sample_config = f"/scenario_generation/configs/config_{config_file_name}.yaml"

max_n_agents = 8  # we will use up to max_n_agents agents only from the data
min_n_agents = 2  # we will discard scenes with less valid agents
min_extent_length = 3.7  # [m] - discard shorter agents
min_extent_width = 1.2  # [m] - discard narrower agents
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
    print(f'Save path {save_dir_path} already exists, will be override...')
else:
    os.mkdir(save_dir_path)

map_data_file_path = Path(save_dir_path, 'map_data').with_suffix('.dat')
save_info_file_path = Path(save_dir_path, 'info').with_suffix('.pkl')
save_data_file_path = Path(save_dir_path, 'data').with_suffix('.h5')

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

saved_mats, dataset_props, labels_hist = process_scenes_data(scene_indices, dataset, dataset_zarr, dm, sim_cfg, cfg,
                                                             min_n_agents, max_n_agents, min_extent_length, min_extent_width,
                                                             verbose=verbose,
                                                             show_html_plot=show_html_plot)
n_scenes = dataset_props['n_scenes']
git_version = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

# ************************************************************
saved_mats_info = {}
with h5py.File(save_data_file_path, 'w') as h5f:
    for var_name, var in saved_mats.items():
        my_ds = h5f.create_dataset(var_name, data=var.data)
        if 'agents' in var_name:
            entity = 'agents'
        else:
            entity = 'map'
        saved_mats_info[var_name] = {'dtype': var.dtype,
                                     'shape': var.shape,
                                     'entity': entity}

with open(save_info_file_path, 'wb') as fid:
    pickle.dump({'dataset_props': dataset_props, 'saved_mats_info': saved_mats_info,
                 'git_version': git_version, 'labels_hist': labels_hist}, fid)

print(f'Saved data of {n_scenes} valid scenes our of {len(scene_indices)} scenes at ', save_dir_path)
