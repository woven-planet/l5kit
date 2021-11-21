import os

import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.data import get_dataset_path
from l5kit.sampling.agent_sampling_vectorized import generate_agent_sample_vectorized
from torch.utils.data.dataloader import default_collate
from l5kit.dataset.utils import move_to_device, move_to_numpy
from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import ClosedLoopSimulator
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from bokeh import plotting


source_dataset_name = "val_data_loader"
scene_indices = [0, 1]
sample_config = "/examples/urban_driver/config.yaml"

########################################################################
# Load data and configurations
########################################################################
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"], project_dir = get_dataset_path()
dm = LocalDataManager(None)
cfg = load_config_data(project_dir + sample_config)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
########################################################################
dataset_cfg = cfg[source_dataset_name]
dataset_zarr = ChunkedDataset(dm.require(dataset_cfg["key"])).open()
vectorizer = build_vectorizer(cfg, dm)
dataset = EgoDatasetVectorized(cfg, dataset_zarr, vectorizer)

print(dataset)

num_simulation_steps = 1
sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                           distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                           start_frame_index=0, show_info=True)

sim_dataset = SimulationDataset.from_dataset_indices(dataset, scene_indices, sim_cfg)
frame_index = 0  # t==0
scene_index_in_list = 0
ego_input = sim_dataset.rasterise_frame_batch(frame_index)[scene_index_in_list]

print(ego_input.keys())
