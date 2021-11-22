import os

import torch
from torch import nn, optim

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.data import get_dataset_path
from l5kit.dataset import EgoDatasetVectorized
from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from l5kit.simulation.dataset import SimulationConfig
from l5kit.vectorization.vectorizer_builder import build_vectorizer


############################################################################################
def inspection(dataset_name="train_data_loader"):
    ########################################################################
    # Load data and configurations
    ########################################################################
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"], project_dir = get_dataset_path()

    dm = LocalDataManager(None)

    cfg = load_config_data(project_dir + "/scenario_generation/config.yaml")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ===== INIT DATASET

    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()

    vectorizer = build_vectorizer(cfg, dm)

    train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)

    ########################################################################
    ## Setup the simulator class to be used to unroll the scene
    ########################################################################
    # ==== DEFINE CLOSED-LOOP SIMULATION
    num_simulation_steps = 50

    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                             distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                             start_frame_index=0, show_info=True)

    weights_scaling = [1.0, 1.0, 1.0]

    _num_predicted_frames = cfg["model_params"]["future_num_frames"]
    _num_predicted_params = len(weights_scaling)

    model = VectorizedUnrollModel(
        history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        num_targets=_num_predicted_params * _num_predicted_frames,
        weights_scaling=weights_scaling,
        criterion=nn.L1Loss(reduction="none"),
        global_head_dropout=cfg["model_params"]["global_head_dropout"],
        disable_other_agents=cfg["model_params"]["disable_other_agents"],
        disable_map=cfg["model_params"]["disable_map"],
        disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
        detach_unroll=cfg["model_params"]["detach_unroll"],
        warmup_num_frames=cfg["model_params"]["warmup_num_frames"],
        discount_factor=cfg["model_params"]["discount_factor"],
    )


    train_cfg = cfg["train_data_loader"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    torch.set_grad_enabled(True)

    scene_index = 3
    scene_dataset = train_dataset.get_scene_dataset(scene_index)
    pass

############################################################################################


if __name__ == "__main__":
    inspection(dataset_name="train_data_loader")
