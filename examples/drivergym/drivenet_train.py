import numpy as np
import torch
from drivenet_eval import eval_model
from stable_baselines3.common import utils
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.kinematic import AckermanPerturbation
from l5kit.planning.model import PlanningModel
from l5kit.random import GaussianRandomGenerator
from l5kit.rasterization import build_rasterizer

dm = LocalDataManager(None)
# get config
cfg = load_config_data("./drivenet_config.yaml")

# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)
mean = np.array([0.0, 0.0, 0.0])  # lateral, longitudinal and angular
std = np.array([0.5, 1.5, np.pi / 6])
perturb_prob = cfg["train_data_loader"]["perturb_probability"]
perturbation = AckermanPerturbation(
    random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=perturb_prob)

# Train Dataset
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = EgoDataset(cfg, train_zarr, rasterizer, perturbation)

# Validation Dataset
eval_cfg = cfg["val_data_loader"]
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
# For evaluation
num_scenes_to_unroll = eval_cfg["max_scene_id"]

# Planning Model
model = PlanningModel(
    model_arch="simple_cnn",
    num_input_channels=rasterizer.num_channels(),
    num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states
    weights_scaling=[1., 1., 1.],
    criterion=nn.MSELoss(reduction="none"),)

# Load train data
train_cfg = cfg["train_data_loader"]
max_train_scene_id = train_cfg["max_scene_id"]
max_train_frame_id = train_dataset.cumulative_sizes[max_train_scene_id]
train_indices = list(range(max_train_frame_id))
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"], sampler=SubsetRandomSampler(train_indices))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train
tr_it = iter(train_dataloader)
max_steps = cfg["train_params"]["max_num_steps"]
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []
model.train()
torch.set_grad_enabled(True)
output_name = cfg["train_params"]["output_name"]
logger = utils.configure_logger(1, "./drivenet_logs/", output_name, True)

for it in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    # Forward pass
    data = {k: v.to(device) for k, v in data.items()}

    result = model(data)
    loss = result["loss"]
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    if (it + 1) % cfg["train_params"]["eval_every_n_steps"] == 0:
        eval_model(model, train_dataset, logger, f"{output_name}_{it}_train_steps", num_scenes_to_unroll)
        eval_model(model, eval_dataset, logger, f"{output_name}_{it}_eval_steps", num_scenes_to_unroll)
        model.train()

    if (it + 1) % cfg["train_params"]["checkpoint_every_n_steps"] == 0:
        to_save = torch.jit.script(model.cpu())
        path_to_save = f"/code/parth/drivenet_models/{output_name}_{it}_steps.pt"
        to_save.save(path_to_save)
        print(f"MODEL STORED at {path_to_save}")
        model = model.to(device)

# Eval
eval_model(model, train_dataset, logger, f"{output_name}_{max_steps}_train_steps", num_scenes_to_unroll)
eval_model(model, eval_dataset, logger, f"{output_name}_{max_steps}_eval_steps", num_scenes_to_unroll)

# model_path = "/code/parth/drivenet_models/dn_h3_p00_1999999_steps.pt"
# model_path = "/code/parth/drivenet_models/dn_h0_p05_run3_1999999_steps.pt"
# model_path = "/code/parth/drivenet_models/dn_h3_p00_run2_1199999_steps.pt"
# model_path = "/code/parth/drivenet_models/dn_h3_p00_run3_1099999_steps.pt"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = torch.load(model_path).to(device)
# model = model.eval()

# eval_model(model, eval_dataset, None, "dn_h3_p00_run3_1099999_eval_steps",
# num_scenes_to_unroll, num_simulation_steps=None)
