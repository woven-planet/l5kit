import os
import pathlib
import random

import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.environment.utils import get_scene_types, get_scene_types_as_dict
from l5kit.kinematic import AckermanPerturbation
from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.random import GaussianRandomGenerator
from l5kit.rasterization import build_rasterizer
from stable_baselines3.common import utils
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from drivenet_eval import eval_model
from drivenet_utils import append_group_index, append_reward_scaling, get_sample_weights, subset_and_subsample
from l5_dro_loss import LossComputer


scene_id_to_type_path = '../../dataset_metadata/validate_turns_metadata.csv'
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./drivenet_config.yaml")

seed = cfg['train_params']['seed']
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

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
cumulative_sizes = train_dataset.cumulative_sizes

if "SCENE_ID_TO_TYPE" not in os.environ:
    raise KeyError("SCENE_ID_TO_TYPE environment variable not set")
scene_id_to_type_mapping_file = os.environ["SCENE_ID_TO_TYPE"]
scene_type_to_id_dict = get_scene_types_as_dict(scene_id_to_type_mapping_file)
scene_id_to_type_list = get_scene_types(scene_id_to_type_mapping_file)
num_groups = len(scene_type_to_id_dict)
group_counts = torch.IntTensor([len(v) for k, v in scene_type_to_id_dict.items()])
group_str = [k for k in scene_type_to_id_dict.keys()]
reward_scale = {"straight": 1.0, "left": 19.5, "right": 16.6}

# Validation Dataset
eval_cfg = cfg["val_data_loader"]
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
# For evaluation
num_scenes_to_unroll = eval_cfg["max_scene_id"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dro_loss_computer = LossComputer(num_groups, group_counts, group_str, device)
# Planning Model
model = RasterizedPlanningModel(
    model_arch="simple_cnn",
    num_input_channels=rasterizer.num_channels(),
    num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states
    weights_scaling=[1., 1., 1.],
    criterion=nn.MSELoss(reduction="none"),
    dro_loss_computer=dro_loss_computer)

# Load train data
train_cfg = cfg["train_data_loader"]
train_scheme = train_cfg["scheme"]
# max_scene_id = train_cfg["max_scene_id"]
num_epochs = train_cfg["epochs"]

# Sub-sample
train_dataset = subset_and_subsample(train_dataset, ratio=train_cfg['ratio'], step=train_cfg['step'])

sampler = None
if train_scheme in {'weighted_sampling', 'group_dro'}:
    sample_weights = get_sample_weights(scene_type_to_id_dict, cumulative_sizes, ratio=train_cfg['ratio'], step=train_cfg['step'])
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_dataset))
    train_cfg["shuffle"] = False

# Reproducibility of Dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"], sampler=sampler, worker_init_fn=seed_worker,
                              generator=g)

model = model.to(device)
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
           lr=5e-4,
           momentum=0.9,
           weight_decay=train_cfg["w_decay"])

if train_cfg["scheduler"] == "one_cycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        max_lr=5e-4,
        pct_start=0.3)
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.1,
        patience=5,
        threshold=0.0001,
        min_lr=0,
        eps=1e-08)

# Train
model.train()
torch.set_grad_enabled(True)

# Logging and Saving
output_name = cfg["train_params"]["output_name"]
save_path = pathlib.Path("./checkpoints/")
save_path.mkdir(parents=True, exist_ok=True)
logger = utils.configure_logger(0, "./drivenet_logs/", output_name, True)

import time
start = time.time()
total_steps = 0
for epoch in range(train_cfg['epochs']):
    for data in tqdm(train_dataloader):
        total_steps += 1
        # Append Reward scaling
        if train_scheme == 'weighted_reward':
            data = append_reward_scaling(data, reward_scale, scene_id_to_type_list)

        # Append Group Index
        if train_scheme == 'group_dro':
            data = append_group_index(data, group_str, scene_id_to_type_list)

        # Forward pass
        data = {k: v.to(device) for k, v in data.items()}

        result = model(data)
        loss = result["loss"]
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.record('rollout/loss', loss.item())
        logger.record('rollout/lr', scheduler.get_last_lr()[0])
        logger.dump(total_steps)

    # Eval
    if (epoch + 1) % cfg["train_params"]["eval_every_n_epochs"] == 0:
        # eval_model(model, train_dataset.dataset, logger, "train", total_steps, num_scenes_to_unroll)
        eval_model(model, eval_dataset, logger, "eval", total_steps, num_scenes_to_unroll,
                   enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_path)
        model.train()

    # Checkpoint
    if (epoch + 1) % cfg["train_params"]["checkpoint_every_n_epochs"] == 0:
        to_save = torch.jit.script(model.cpu())
        path_to_save = f"./checkpoints/{output_name}_{total_steps}_steps.pt"
        to_save.save(path_to_save)
        model = model.to(device)

print("Time: ", time.time() - start)

# Final Eval
# eval_model(model, train_dataset.dataset, logger, "train", total_steps, num_scenes_to_unroll)
eval_model(model, eval_dataset, logger, "eval", total_steps, num_scenes_to_unroll,
           enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_path)

# Final Checkpoint
to_save = torch.jit.script(model.cpu())
path_to_save = f"./checkpoints/{output_name}_{total_steps}_steps.pt"
to_save.save(path_to_save)
model = model.to(device)
