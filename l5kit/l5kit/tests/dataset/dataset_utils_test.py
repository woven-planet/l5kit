import pytest

from l5kit.data import ChunkedDataset, LocalDataManager, get_frames_slice_from_scenes
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer


@pytest.mark.parametrize("scene_idx", [0, pytest.param(999, marks=pytest.mark.xfail)])
def test_get_scene_indices_ego(scene_idx: int, zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict) -> None:
    rast_params = cfg["raster_params"]
    rast_params["map_type"] = "box_debug"
    rasterizer = build_rasterizer(cfg, dmg)

    dataset = EgoDataset(cfg, zarr_dataset, rasterizer)
    scene_indices = dataset.get_scene_indices(scene_idx)
    frame_slice = get_frames_slice_from_scenes(zarr_dataset.scenes[scene_idx])
    assert scene_indices[0] == frame_slice.start
    assert scene_indices[-1] == frame_slice.stop - 1
