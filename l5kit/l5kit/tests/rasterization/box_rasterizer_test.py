import numpy as np
import pytest

from l5kit.data import AGENT_DTYPE, ChunkedDataset, filter_agents_by_frames, LocalDataManager
from l5kit.rasterization import build_rasterizer
from l5kit.rasterization.box_rasterizer import draw_boxes, get_box_world_coords


def test_empty_boxes() -> None:
    # naive test with empty arrays
    agents = np.empty(0, dtype=AGENT_DTYPE)
    to_image_space = np.eye(3)
    im = draw_boxes((200, 200), to_image_space, agents, color=255)
    assert im.sum() == 0


def test_draw_boxes() -> None:
    centroid_1 = (90, 100)
    centroid_2 = (150, 160)

    agents = np.zeros(2, dtype=AGENT_DTYPE)
    agents[0]["extent"] = (20, 20, 20)
    agents[0]["centroid"] = centroid_1

    agents[1]["extent"] = (20, 20, 20)
    agents[1]["centroid"] = centroid_2

    to_image_space = np.eye(3)
    im = draw_boxes((200, 200), to_image_space, agents, color=1)

    # due to subpixel precision we can't check the exact number of pixels
    # check that a 10x10 centred on the boxes is all 1
    assert np.allclose(im[centroid_1[1] - 5: centroid_1[1] + 5, centroid_1[0] - 5: centroid_1[0] + 5], 1)
    assert np.allclose(im[centroid_2[1] - 5: centroid_2[1] + 5, centroid_2[0] - 5: centroid_2[0] + 5], 1)


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
    out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1], [])
    assert out[..., -1].sum() > 0


def test_agents_layer_out(hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"

    cfg["raster_params"]["filter_agents_threshold"] = 1.0
    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1], [])
    assert out[..., 0].sum() == 0

    cfg["raster_params"]["filter_agents_threshold"] = 0.0
    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1], [])
    assert out[..., 0].sum() > 0


def test_agent_as_ego(hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"
    cfg["raster_params"]["filter_agents_threshold"] = -1  # take everything
    rasterizer = build_rasterizer(cfg, dmg)

    agents = hist_data[1][0]
    for ag in agents:
        out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1], [], ag)
        assert out[..., -1].sum() > 0


@pytest.mark.parametrize("render_ego_history", [True, False])
def test_render_ego_history(hist_data: tuple, dmg: LocalDataManager, cfg: dict, render_ego_history: bool) -> None:
    hist_length = 5
    cfg["raster_params"]["map_type"] = "box_debug"
    cfg["model_params"]["history_num_frames"] = hist_length
    cfg["model_params"]["render_ego_history"] = render_ego_history

    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][: hist_length + 1], hist_data[1][: hist_length + 1], [])
    ego_indices = range(hist_length + 1, out.shape[-1])
    for ego_idx in ego_indices:
        if render_ego_history:
            assert out[..., ego_idx].sum() > 0
        else:
            if ego_idx == ego_indices.start:
                assert out[..., ego_idx].sum() > 0
            else:
                assert out[..., ego_idx].sum() == 0


def test_out_shape(hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    hist_length = 5
    cfg["raster_params"]["map_type"] = "box_debug"
    cfg["model_params"]["history_num_frames"] = hist_length

    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][: hist_length + 1], hist_data[1][: hist_length + 1], [])
    assert out.shape == (224, 224, (hist_length + 1) * 2)


def test_box_world_coords_empty() -> None:
    agents = np.zeros(0, dtype=AGENT_DTYPE)
    out = get_box_world_coords(agents)
    assert len(out) == 0


def test_box_world_coords() -> None:
    agents = np.zeros(4, dtype=AGENT_DTYPE)
    # the first agents has everything at 0
    # the second is a translated square
    agents[1]["extent"] = (4, 4, 4)
    agents[1]["centroid"] = (10, 10)
    # the third one is a rectangle rotated by 90 degrees
    agents[2]["extent"] = (2, 4, 2)
    agents[2]["yaw"] = np.radians(90)
    agents[2]["centroid"] = (10, 10)
    # the third one is a rectangle rotated by 270 degrees
    agents[3]["extent"] = (2, 4, 2)
    agents[3]["yaw"] = np.radians(270)
    agents[3]["centroid"] = (10, 10)

    out = get_box_world_coords(agents)
    assert len(out) == 4
    assert np.allclose(out[0], 0.)
    assert np.allclose(out[1], np.asarray([[8, 8], [8, 12], [12, 12], [12, 8]]))

    expected_points = np.asarray([[8, 9], [8, 11], [12, 11], [12, 9]])
    for out_box in out[2:4]:
        for exp_point in expected_points:
            assert np.any(np.linalg.norm(out_box - exp_point, axis=-1) < 1e5)
