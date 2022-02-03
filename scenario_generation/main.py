import os
import subprocess
import pickle
import numpy as np
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from general_util import get_dataset_path
from l5kit.dataset import EgoDatasetVectorized
from l5kit.simulation.dataset import SimulationConfig
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from scenario_generation.extract_scenario_dataset import process_scenes_data

########################################################################
verbose = 1  # 0 | 1
show_html_plot = False
config_file_name = 'sample'  # 'sample' | 'train' | 'train_full'
source_name = "train_data_loader"  # "train_data_loader | "val_data_loader"
save_dir_name = 'l5kit_data_' + config_file_name + '_' + source_name
sample_config = f"/scenario_generation/configs/config_{config_file_name}.yaml"
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

########################################################################
# Load data and configurations
########################################################################
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"], project_dir = get_dataset_path()
dm = LocalDataManager(None)
cfg = load_config_data(project_dir + sample_config)

dataset_cfg = cfg[source_name]

dataset_zarr = ChunkedDataset(dm.require(dataset_cfg["key"])).open()
n_scenes = len(dataset_zarr.scenes)
vectorizer = build_vectorizer(cfg, dm)
dataset = EgoDatasetVectorized(cfg, dataset_zarr, vectorizer)

print(dataset)
print(f'Dataset source: {cfg[source_name]["key"]}, number of scenes total: {n_scenes}')

num_simulation_steps = 10
sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                           distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                           start_frame_index=0, show_info=True)

scene_indices = list(range(n_scenes))
# scene_indices = [39]

agents_feat, map_feat, agent_types_labels, labels_hist = process_scenes_data(scene_indices, dataset, dataset_zarr,
                                                                             dm, sim_cfg, cfg,
                                                                             verbose=verbose, show_html_plot=show_html_plot)

git_version = subprocess.check_output(["git", "describe", "--always"]).strip().decode()


#
# np.savez(map_data_file_path,
#          map_elems_points=map_feat['map_elems_points'],
#          map_elems_n_points_orig=map_feat['map_elems_n_points_orig'],
#          map_elems_exists=map_feat['map_elems_exists'],
#          )

for var_name, var in map_feat.items():
    save_file_path = os.path.join(save_dir_path, var_name)
    # Create a memmap with dtype and shape that matches our data:
    fp = np.memmap(save_file_path, dtype=var.dtype, mode='w+', shape=var.shape)
    fp[:] = var[:]  # write data to memmap array
    fp.flush()  # Flushes memory changes to disk in order to read them back

with open(info_file_path, 'wb') as fid:
    pickle.dump({'agents_feat': agents_feat,  'agent_types_labels': agent_types_labels,
                 'git_version': git_version, 'labels_hist': labels_hist}, fid)


print(f'Saved data of {len(scene_indices)} scenes at ', save_dir_path)
