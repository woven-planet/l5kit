from typing import Callable

import pytest

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset, build_dataloader
from l5kit.rasterization import build_rasterizer


@pytest.mark.parametrize("scene_indices", [(-1,), (0,), pytest.param((0, 1000), marks=pytest.mark.xfail)])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_build_dataloader(scene_indices: tuple, dataset_cls: Callable) -> None:
    cfg = load_config_data("./l5kit/tests/artefacts/config.yaml")
    cfg["train_data_loader"]["datasets"][0]["scene_indices"] = scene_indices
    dm = LocalDataManager("./l5kit/tests/artefacts/")
    rasterizer = build_rasterizer(cfg, dm)
    dl = build_dataloader(cfg, "train", dm, dataset_cls, rasterizer)
    next(iter(dl))
