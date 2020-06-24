from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision.models.resnet import resnet50
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.dataset.dataloader_builder import build_dataloader
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_coords_as_csv, compute_mse_error_csv
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable

import os


def build_model(cfg: Dict) -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = resnet50(pretrained=True)

    # change input size
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size
    # X, Y  * number of future states
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model


def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    targets = data["target_positions"].to(device).reshape(len(data["target_positions"]), -1)
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss.mean()
    return loss, outputs


def main():
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = "/tmp/l5kit_data"
    # get config
    cfg = load_config_data("./agent_motion_config.yaml")
    print(cfg)

    dm = LocalDataManager(None)
    # ===== INIT DATASETS
    rasterizer = build_rasterizer(cfg, dm)
    train_dataloader = build_dataloader(cfg, "train", dm, AgentDataset, rasterizer)
    eval_dataloader = build_dataloader(cfg, "val", dm, AgentDataset, rasterizer)

    # ==== INIT MODEL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="none")

    # ==== TRAIN LOOP
    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    losses_train = []
    for _ in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)

        model.train()
        torch.set_grad_enabled(True)
        loss, _ = forward(data, model, device, criterion)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    # ==== EVAL LOOP
    model.eval()
    torch.set_grad_enabled(False)
    losses_eval = []

    # store information for evaluation
    future_coords_offsets_pd = []
    future_coords_offsets_gt = []

    timestamps = []
    agent_ids = []
    progress_bar = tqdm(eval_dataloader)
    for data in progress_bar:
        loss, ouputs = forward(data, model, device, criterion)
        losses_eval.append(loss.item())
        progress_bar.set_description(f"Running EVAL, loss: {loss.item()} loss(avg): {np.mean(losses_eval)}")

        future_coords_offsets_pd.append(ouputs.reshape(len(ouputs), -1, 2).cpu().numpy())
        future_coords_offsets_gt.append(data["target_positions"].reshape(len(ouputs), -1, 2).cpu().numpy())

        timestamps.append(data["timestamp"].numpy())
        agent_ids.append(data["track_id"].numpy())

        # ==== COMPUTE CSV
    pred_path = f"{gettempdir()}/pred.csv"
    gt_path = f"{gettempdir()}/gt.csv"

    write_coords_as_csv(pred_path, future_num_frames=cfg["model_params"]["future_num_frames"],
                        future_coords_offsets=np.concatenate(future_coords_offsets_pd),
                        timestamps=np.concatenate(timestamps),
                        agent_ids=np.concatenate(agent_ids))
    write_coords_as_csv(gt_path, future_num_frames=cfg["model_params"]["future_num_frames"],
                        future_coords_offsets=np.concatenate(future_coords_offsets_gt),
                        timestamps=np.concatenate(timestamps),
                        agent_ids=np.concatenate(agent_ids))
    # get a pretty visualisation of the errors
    table = PrettyTable(field_names=["future step", "MSE"])
    table.float_format = ".2"
    steps = range(1, cfg["model_params"]["future_num_frames"] + 1)
    errors = compute_mse_error_csv(gt_path, pred_path)
    for step_idx, step_mse in zip(steps, errors):
        table.add_row([step_idx, step_mse])
    print(table)


if __name__ == "__main__":
    main()
