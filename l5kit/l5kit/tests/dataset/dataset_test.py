from typing import Callable

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset, Subset

from l5kit.configs import load_config_data
from l5kit.data import ChunkedStateDataset, load_pose_to_ecef
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import (
    BoxRasterizer,
    Rasterizer,
    SatBoxRasterizer,
    SatelliteRasterizer,
    SemanticRasterizer,
    SemBoxRasterizer,
    StubRasterizer,
)


@pytest.fixture(scope="module")
def zarr_dataset() -> ChunkedStateDataset:
    zarr_dataset = ChunkedStateDataset(path="./l5kit/tests/data/single_scene.zarr")
    zarr_dataset.open()
    return zarr_dataset


def get_rasterizer(rast_name: str, cfg: dict) -> Rasterizer:
    # PARAMS
    # sat params
    # TODO replace when we have the map in l5kit
    map_to_sat = np.block(
        [[np.eye(3) / 100, np.asarray([[1000], [1000], [1]])], [np.asarray([[0, 0, 0, 1]])]]
    )  # just a translation and scale
    map_im = np.zeros((10000, 10000, 3), dtype=np.uint8)
    # sem params
    pose_to_ecef = load_pose_to_ecef()
    semantic_map = {
        "lanes": [],
        "lat": 37.44,
        "lon": 122.14,
        "lanes_bounds": {
            "min_x": np.asarray([]),
            "max_x": np.asarray([]),
            "min_y": np.asarray([]),
            "max_y": np.asarray([]),
        },
        "crosswalks": [],
        "crosswalks_bounds": {
            "min_x": np.asarray([]),
            "max_x": np.asarray([]),
            "min_y": np.asarray([]),
            "max_y": np.asarray([]),
        },
    }

    raster_size = cfg["raster_params"]["raster_size"]
    pixel_size = np.asarray(cfg["raster_params"]["pixel_size"])
    ego_center = np.asarray(cfg["raster_params"]["ego_center"])

    if rast_name == "box":
        return BoxRasterizer(raster_size, pixel_size, ego_center, filter_agents_threshold=-1,)
    elif rast_name == "sat":
        return SatelliteRasterizer(raster_size, pixel_size, ego_center, map_im=map_im, map_to_sat=map_to_sat,)
    elif rast_name == "satbox":
        return SatBoxRasterizer(
            raster_size, pixel_size, ego_center, filter_agents_threshold=-1, map_im=map_im, map_to_sat=map_to_sat,
        )
    elif rast_name == "sem":
        return SemanticRasterizer(
            raster_size, pixel_size, ego_center, semantic_map=semantic_map, pose_to_ecef=pose_to_ecef,
        )
    elif rast_name == "sembox":
        return SemBoxRasterizer(
            raster_size,
            pixel_size,
            ego_center,
            semantic_map=semantic_map,
            pose_to_ecef=pose_to_ecef,
            filter_agents_threshold=-1,
        )
    else:
        raise NotImplementedError


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


@pytest.mark.parametrize("rast_name", ["box", "sat", "satbox", "sem", "sembox"])  # TODO others params?
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_dataset_rasterizer(rast_name: str, dataset_cls: Callable, zarr_dataset: ChunkedStateDataset) -> None:
    cfg = load_config_data("./l5kit/configs/default.yaml")
    # replace th for agents for AgentDataset test
    cfg["raster_params"]["filter_agents_threshold"] = 0.5

    rasterizer = get_rasterizer(rast_name, cfg)
    dataset = dataset_cls(cfg=cfg, zarr_dataset=zarr_dataset, rasterizer=rasterizer, perturbation=None)
    check_sample(cfg, dataset)
    check_torch_loading(dataset)


@pytest.mark.parametrize("frame_idx", [0, 10, 774, pytest.param(775, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_frame_index_interval(dataset_cls: Callable, frame_idx: int, zarr_dataset: ChunkedStateDataset) -> None:
    cfg = load_config_data("./l5kit/configs/default.yaml")
    # replace th for agents for AgentDataset test
    cfg["raster_params"]["filter_agents_threshold"] = 0.5
    rasterizer = StubRasterizer((100, 100), np.asarray((0.25, 0.25)), np.asarray((0.5, 0.5)), 0)
    dataset = dataset_cls(cfg, zarr_dataset, rasterizer, None)
    indices = dataset.get_frame_indices(frame_idx)
    subdata = Subset(dataset, indices)
    for _ in subdata:  # type: ignore
        pass


@pytest.mark.parametrize("scene_idx", [0, pytest.param(1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_scene_index_interval(dataset_cls: Callable, scene_idx: int, zarr_dataset: ChunkedStateDataset) -> None:
    cfg = load_config_data("./l5kit/configs/default.yaml")
    # replace th for agents for AgentDataset test
    cfg["raster_params"]["filter_agents_threshold"] = 0.5
    rasterizer = StubRasterizer((100, 100), np.asarray((0.25, 0.25)), np.asarray((0.5, 0.5)), 0)
    dataset = dataset_cls(cfg, zarr_dataset, rasterizer, None)
    indices = dataset.get_scene_indices(scene_idx)
    subdata = Subset(dataset, indices)
    for _ in subdata:  # type: ignore
        pass
