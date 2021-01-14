from l5kit.data import ChunkedDataset, filter_agents_by_frames, LocalDataManager
from l5kit.rasterization import build_rasterizer


def test_shape(zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict) -> None:
    hist_length = 10
    cfg["raster_params"]["map_type"] = "py_satellite"
    cfg["raster_params"]["filter_agents_threshold"] = 1.0
    cfg["model_params"]["history_num_frames"] = hist_length

    rasterizer = build_rasterizer(cfg, dmg)
    frames = zarr_dataset.frames[: hist_length + 1][::-1]
    agents = filter_agents_by_frames(frames, zarr_dataset.agents)
    out = rasterizer.rasterize(frames, agents, [])  # TODO TR_FACES
    assert out.shape == (224, 224, (hist_length + 1) * 2 + 3)
