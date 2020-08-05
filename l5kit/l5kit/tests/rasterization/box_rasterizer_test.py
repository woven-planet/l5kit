import numpy as np
import pytest

from l5kit.data import AGENT_DTYPE, ChunkedDataset, LocalDataManager, filter_agents_by_frames
from l5kit.rasterization import build_rasterizer
from l5kit.rasterization.box_rasterizer import draw_boxes


def test_empty_boxes() -> None:
    # naive test with empty arrays
    agents = np.empty(0, dtype=AGENT_DTYPE)
    to_image_space = np.eye(3)
    im = draw_boxes((200, 200), to_image_space, agents, color=255)
    assert im.sum() == 0


def test_draw_boxes() -> None:
    agents = np.zeros(2, dtype=AGENT_DTYPE)
    agents[0]["extent"] = (20, 20, 20)
    agents[0]["centroid"] = (100, 100)

    agents[1]["extent"] = (20, 20, 20)
    agents[1]["centroid"] = (150, 150)

    to_image_space = np.eye(3)
    im = draw_boxes((200, 200), to_image_space, agents, color=1)
    assert im.sum() == (21 * 21) * 2


@pytest.fixture(scope="module")
def hist_data(zarr_dataset: ChunkedDataset) -> tuple:
    hist_frames = zarr_dataset.frames[100:111][::-1]  # reverse to get them as history
    hist_agents = filter_agents_by_frames(hist_frames, zarr_dataset.agents)
    return hist_frames, hist_agents


@pytest.mark.parametrize("ego_center", [(0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75)])
def test_ego_layer_out_center_configs(ego_center: tuple, hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"
    cfg["raster_params"]["ego_center"] = np.asarray(ego_center)

    rasterizer = build_rasterizer(cfg, dmg)
    out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1])
    assert out[..., -1].sum() > 0


def test_agents_layer_out(hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"

    cfg["raster_params"]["filter_agents_threshold"] = 1.0
    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1])
    assert out[..., 0].sum() == 0

    cfg["raster_params"]["filter_agents_threshold"] = 0.0
    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1])
    assert out[..., 0].sum() > 0


def test_agent_as_ego(hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"
    cfg["raster_params"]["filter_agents_threshold"] = -1  # take everything
    rasterizer = build_rasterizer(cfg, dmg)

    agents = hist_data[1][0]
    for ag in agents:
        out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1], ag)
        assert out[..., -1].sum() > 0


def test_out_shape(hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    hist_length = 5
    cfg["raster_params"]["map_type"] = "box_debug"
    cfg["model_params"]["history_num_frames"] = hist_length

    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][: hist_length + 1], hist_data[1][: hist_length + 1])
    assert out.shape == (224, 224, (hist_length + 1) * 2)
