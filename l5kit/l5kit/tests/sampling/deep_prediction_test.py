import functools
from typing import Callable

import numpy as np
import pytest

from l5kit.data import AGENT_DTYPE, ChunkedDataset, FRAME_DTYPE
from l5kit.rasterization import RenderContext, StubRasterizer
from l5kit.sampling.agent_sampling import generate_agent_sample


def get_partial(cfg: dict, history_num_frames: int, future_num_frames: int, step_time: float, ) -> Callable:
    rast_params = cfg["raster_params"]

    render_context = RenderContext(
        raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
        pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
        center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
        set_origin_to_bottom=cfg["raster_params"]["set_origin_to_bottom"],
    )

    rasterizer = StubRasterizer(render_context)
    return functools.partial(
        generate_agent_sample,
        render_context=render_context,
        history_num_frames=history_num_frames,
        future_num_frames=future_num_frames,
        step_time=step_time,
        filter_agents_threshold=rast_params["filter_agents_threshold"],
        rasterizer=rasterizer,
    )


def test_no_frames(zarr_dataset: ChunkedDataset, cfg: dict) -> None:
    gen_partial = get_partial(cfg, 2, 4, 0.1)
    with pytest.raises(IndexError):
        gen_partial(
            state_index=0,
            frames=np.zeros(0, FRAME_DTYPE),
            agents=np.zeros(0, AGENT_DTYPE),
            tl_faces=np.zeros(0),  # TODO TL_FACES
            selected_track_id=None,
        )


def test_out_bounds(zarr_dataset: ChunkedDataset, cfg: dict) -> None:
    gen_partial = get_partial(cfg, 0, 10, 0.1)
    data = gen_partial(
        state_index=0,
        frames=np.asarray(zarr_dataset.frames[90:96]),
        agents=zarr_dataset.agents,
        tl_faces=np.zeros(0),  # TODO TL_FACES
        selected_track_id=None,
    )
    assert bool(np.all(data["target_availabilities"][:5])) is True
    assert bool(np.all(data["target_availabilities"][5:])) is False


def test_future(zarr_dataset: ChunkedDataset, cfg: dict) -> None:
    steps = 1, 2, 4  # all of these should work
    for step in steps:
        gen_partial = get_partial(cfg, 2, step, 0.1)
        data = gen_partial(
            state_index=10,
            frames=np.asarray(zarr_dataset.frames[90:150]),
            agents=zarr_dataset.agents,
            tl_faces=np.zeros(0),  # TODO TL_FACES
            selected_track_id=None,
        )
        assert data["target_positions"].shape == (step, 2)
        assert data["target_yaws"].shape == (step, 1)
        assert data["target_availabilities"].shape == (step,)
        assert data["centroid"].shape == (2,)
        assert isinstance(data["yaw"], float)
        assert data["extent"].shape == (3,)
        assert bool(np.all(data["target_availabilities"])) is True
