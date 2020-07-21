from typing import Callable

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset, Subset

from l5kit.configs import load_config_data
from l5kit.data import ChunkedStateDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import StubRasterizer, build_rasterizer


@pytest.fixture(scope="module")
def zarr_dataset() -> ChunkedStateDataset:
    zarr_dataset = ChunkedStateDataset(path="./l5kit/tests/artefacts/single_scene.zarr")
    zarr_dataset.open()
    return zarr_dataset


def check_sample(cfg: dict, dataset: Dataset) -> None:
    iterator = iter(dataset)  # type: ignore
    for i in range(10):
        el = next(iterator)
        assert el["image"].shape[1:] == tuple(cfg["raster_params"]["raster_size"])
        assert len(el["target_positions"]) == cfg["model_params"]["future_num_frames"]
        assert len(el["target_yaws"]) == cfg["model_params"]["future_num_frames"]
        assert len(el["target_availabilities"]) == cfg["model_params"]["future_num_frames"]
        assert el["world_to_image"].shape == (3, 3)


def check_torch_loading(dataset: Dataset) -> None:
    # test dataloader
    iterator = iter(DataLoader(dataset, batch_size=4))  # type: ignore
    next(iterator)


@pytest.mark.parametrize("rast_name", ["py_satellite", "py_semantic", "box_debug", "satellite_debug"])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_dataset_rasterizer(rast_name: str, dataset_cls: Callable, zarr_dataset: ChunkedStateDataset) -> None:
    cfg = load_config_data("./l5kit/tests/artefacts/config.yaml")
    dm = LocalDataManager("./l5kit/tests/artefacts/")
    rasterizer = build_rasterizer(cfg, dm)
    dataset = dataset_cls(cfg=cfg, zarr_dataset=zarr_dataset, rasterizer=rasterizer, perturbation=None)
    check_sample(cfg, dataset)
    check_torch_loading(dataset)


@pytest.mark.parametrize("frame_idx", [0, 10, 774, pytest.param(775, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_frame_index_interval(dataset_cls: Callable, frame_idx: int, zarr_dataset: ChunkedStateDataset) -> None:
    cfg = load_config_data("./l5kit/tests/artefacts/config.yaml")
    rasterizer = StubRasterizer((100, 100), np.asarray((0.25, 0.25)), np.asarray((0.5, 0.5)), 0)
    dataset = dataset_cls(cfg, zarr_dataset, rasterizer, None)
    indices = dataset.get_frame_indices(frame_idx)
    subdata = Subset(dataset, indices)
    for _ in subdata:  # type: ignore
        pass


@pytest.mark.parametrize("scene_idx", [0, pytest.param(1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_scene_index_interval(dataset_cls: Callable, scene_idx: int, zarr_dataset: ChunkedStateDataset) -> None:
    cfg = load_config_data("./l5kit/tests/artefacts/config.yaml")
    rasterizer = StubRasterizer((100, 100), np.asarray((0.25, 0.25)), np.asarray((0.5, 0.5)), 0)
    dataset = dataset_cls(cfg, zarr_dataset, rasterizer, None)
    indices = dataset.get_scene_indices(scene_idx)
    subdata = Subset(dataset, indices)
    for _ in subdata:  # type: ignore
        pass


@pytest.mark.parametrize("history_num_frames", [1, 2, 3, 4])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_non_zero_history(history_num_frames: int, dataset_cls: Callable, zarr_dataset: ChunkedStateDataset) -> None:
    cfg = load_config_data("./l5kit/tests/artefacts/config.yaml")
    cfg["model_params"]["history_num_frames"] = history_num_frames
    rast_params = cfg["raster_params"]
    rast_params["map_type"] = "box_debug"
    dm = LocalDataManager("./l5kit/tests/artefacts/")
    rasterizer = build_rasterizer(cfg, dm)

    dataset = dataset_cls(cfg, zarr_dataset, rasterizer, None)
    indexes = [0, 1, 10, -1]  # because we pad, even the first index should have an (entire black) history
    for idx in indexes:
        data = dataset[idx]
        assert data["image"].shape == (2 * (history_num_frames + 1), *rast_params["raster_size"])
