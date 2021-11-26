import os
from pathlib import Path
import numpy as np
import pickle
import torch
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.data import get_dataset_path
from l5kit.dataset import EgoDatasetVectorized
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from extract_scenario_dataset import get_scenes_batch


########################################################################
source_dataset_name = "train_data_loader"
sample_config = "/scenario_generation/configs/config_train_full.yaml"   # config_train_full.yaml"
saved_file_name = 'l5kit_train_full'
# Our changes:
# max_retrieval_distance_m: 60
# train_data_loader:  key: "scenes/train.zarr"

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

num_simulation_steps = 10
sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                           distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                           start_frame_index=0, show_info=True)

# scene_indices = [48, 49]
scene_indices = list(range(n_scenes))

agents_feat, map_feat = get_scenes_batch(scene_indices, dataset, dataset_zarr, dm, sim_cfg, cfg, verbose=0)


save_file_path = saved_file_name + '.pkl'
with open(save_file_path, 'wb') as fid:
    pickle.dump([agents_feat, map_feat], fid)
print(f'Saved data of {len(scene_indices)} scenes at ', save_file_path)
