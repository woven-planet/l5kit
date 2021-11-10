
import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable
import os

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.visualization.visualizer.zarr_utils import zarr_to_visualizer_scene
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI

# set env variable for data
root_project_dir = os.path.dirname(os.getcwd())
os.environ["L5KIT_DATA_FOLDER"] = open(root_project_dir + "/examples/dataset_dir.txt", "r").read().strip()
dm = LocalDataManager(None)
# get config
cfg = load_config_data(root_project_dir + "/examples/urban_driver/config.yaml")

# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()

vectorizer = build_vectorizer(cfg, dm)
train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)
print(train_dataset)

# Visualisation Examples
output_notebook()
mapAPI = MapAPI.from_cfg(dm, cfg)
for scene_idx in range(4):
    out = zarr_to_visualizer_scene(train_zarr.get_scene_dataset(scene_idx), mapAPI)
    out_vis = visualize(scene_idx, out)
    show(out_vis)
plt.show()

