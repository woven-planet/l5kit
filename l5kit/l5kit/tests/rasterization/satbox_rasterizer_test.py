import pytest

from l5kit.data import ChunkedDataset, LocalDataManager, filter_agents_by_frames
from l5kit.rasterization import build_rasterizer


@pytest.fixture(scope="module")
def dataset() -> ChunkedDataset:
    zarr_dataset = ChunkedDataset(path="./l5kit/tests/artefacts/single_scene.zarr")
    zarr_dataset.open()
    return zarr_dataset


def test_shape(dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict) -> None:
    hist_length = 10
    cfg["raster_params"]["map_type"] = "py_satellite"
    cfg["raster_params"]["filter_agents_threshold"] = 1.0
    cfg["model_params"]["history_num_frames"] = hist_length

    rasterizer = build_rasterizer(cfg, dmg)

    frames = dataset.frames[: hist_length + 1][::-1]
    agents = filter_agents_by_frames(frames, dataset.agents)
    out = rasterizer.rasterize(frames, agents)
    assert out.shape == (224, 224, (hist_length + 1) * 2 + 3)
