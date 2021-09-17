import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDatasetVectorized
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator
from l5kit.data.map_api import CACHE_SIZE, InterpolationMethod, MapAPI
from l5kit.rasterization.rasterizer_builder import get_hardcoded_world_to_ecef
from l5kit.vectorization.vectorizer import Vectorizer
import os
from l5kit.vectorization.vectorizer_builder import build_vectorizer

# TODO: hacky (temp) run file

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/tmp/l5kit_data"
dm = LocalDataManager(None)

# get config 
cfg = load_config_data("/code/l5kit/l5kit/l5kit/dev/config_olt.yaml")

# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
# rasterisation and perturbation

# TODO
rasterizer = None # build_rasterizer(cfg, dm)

vectorizer = build_vectorizer(cfg, dm)

train_dataset = EgoDatasetVectorized(cfg, train_zarr, rasterizer, vectorizer)

weights_scaling = [1.0, 1.0, 1.0]

_num_predicted_frames = cfg["model_params"]["future_num_frames"]
_num_predicted_params = len(weights_scaling)

model = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=_num_predicted_params * _num_predicted_frames,
            weights_scaling=weights_scaling,
            criterion=nn.L1Loss(reduction="none"),
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
        )

train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


tr_it = iter(train_dataloader)
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []
model.train()
torch.set_grad_enabled(True)

for _ in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    # Forward pass
    data = {k: v.to(device) for k, v in data.items()}
    for key in data:
        print(f"{key}: {data[key].shape} - {data[key].dtype}")
    import ipdb
    ipdb.set_trace()
    result = model(data)
    loss = result["loss"]
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")