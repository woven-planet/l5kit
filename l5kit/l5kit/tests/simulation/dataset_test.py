
import pytest

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.simulation.dataset import SimulationDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from pathlib import Path
import numpy as np
import torch
from l5kit.geometry import rotation33_as_yaw


def test_simulation_ego(zarr_cat_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict, tmp_path: Path) -> None:
    rasterizer = build_rasterizer(cfg, dmg)

    scene_indices = list(range(len(zarr_cat_dataset.scenes)))

    ego_dataset = EgoDataset(cfg, zarr_cat_dataset, rasterizer)
    dataset = SimulationDataset(ego_dataset, scene_indices)

    # this also ensure order is checked
    assert list(dataset.scene_dataset_batch.keys()) == scene_indices

    # ensure we can call the getitem
    out_0 = dataset[(0, 0)]
    assert len(out_0) > 0
    out_last = dataset[(0, len(dataset) - 1)]
    assert len(out_last) > 0
    with pytest.raises(IndexError):
        _ = dataset[(0, len(dataset))]

    # ensure we can call the aggregated get frame
    out_0 = dataset.rasterise_frame_batch(0)
    assert len(out_0) == len(scene_indices)
    out_last = dataset.rasterise_frame_batch(len(dataset) - 1)
    assert len(out_last) == len(scene_indices)
    with pytest.raises(IndexError):
        _ = dataset.rasterise_frame_batch(len(dataset))

    # ensure we can set the ego in multiple frames for all scenes
    frame_indices = np.random.randint(0, len(dataset), 10)
    for frame_idx in frame_indices:
        mock_tr = torch.rand(len(scene_indices), 12, 2)
        mock_yaw = torch.rand(len(scene_indices), 12)

        dataset.set_ego_for_frame(frame_idx, 0, mock_tr, mock_yaw)

        for scene_idx in scene_indices:
            scene_zarr = dataset.scene_dataset_batch[scene_idx].dataset
            ego_tr = scene_zarr.frames["ego_translation"][frame_idx]
            ego_yaw = rotation33_as_yaw(scene_zarr.frames["ego_rotation"][frame_idx])

            assert np.allclose(mock_tr[scene_idx, 0].numpy(), ego_tr[:2])
            assert np.allclose(mock_yaw[scene_idx, 0].numpy(), ego_yaw)


def test_simulation_agents(zarr_cat_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict, tmp_path: Path) -> None:
    rasterizer = build_rasterizer(cfg, dmg)

    scene_indices = list(range(len(zarr_cat_dataset.scenes)))

    ego_dataset = EgoDataset(cfg, zarr_cat_dataset, rasterizer)
    dataset = SimulationDataset(ego_dataset, scene_indices)
    # TODO: implement here
