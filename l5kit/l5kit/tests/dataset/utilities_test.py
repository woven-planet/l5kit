from typing import Callable

import pytest

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset, build_dataloader

from .dataset_test import get_rasterizer  # TODO remove once we can instantiate rasterizers


@pytest.mark.parametrize("scene_indices", [(-1,), (0,), pytest.param((0, 1000), marks=pytest.mark.xfail)])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_build_dataloader(scene_indices: tuple, dataset_cls: Callable) -> None:
    cfg = load_config_data("./l5kit/configs/default.yaml")

    # replace in cfg to point to the test dataset and seqs
    cfg["train_data_loader"]["datasets"] = [
        {"key": "./l5kit/tests/data/single_scene.zarr", "scene_indices": scene_indices}
    ]
    cfg["train_data_loader"]["num_workers"] = 0
    # replace th for agents for AgentDataset test
    cfg["raster_params"]["filter_agents_threshold"] = 0.5
    dl = build_dataloader(cfg, "train", LocalDataManager("."), dataset_cls, get_rasterizer("box", cfg), None)
    next(iter(dl))
