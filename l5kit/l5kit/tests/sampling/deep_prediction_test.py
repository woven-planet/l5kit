import functools
import unittest
from typing import Callable

import numpy as np

from l5kit.data import AGENT_DTYPE, FRAME_DTYPE, ChunkedStateDataset
from l5kit.rasterization import StubRasterizer
from l5kit.sampling import generate_agent_sample


class TestDeepPredictionSampling(unittest.TestCase):
    def __init__(self, *args, **kwargs):  # type: ignore
        super(TestDeepPredictionSampling, self).__init__(*args, **kwargs)
        self.dataset = ChunkedStateDataset(path="./l5kit/tests/data/single_scene.zarr")
        self.dataset.open()

        self.raster_size = (100, 100)
        self.pixel_size = np.array([1.0, 1.0])
        self.ego_center = np.array([0.5, 0.25])
        self.filter_agents_threshold = 0.5
        self.rast = StubRasterizer(self.raster_size, self.pixel_size, self.ego_center, self.filter_agents_threshold)

    def get_partial(
        self, history_num_frames: int, history_step_size: int, future_num_frames: int, future_step_size: int
    ) -> Callable:
        return functools.partial(
            generate_agent_sample,
            raster_size=self.raster_size,
            pixel_size=self.pixel_size,
            ego_center=self.ego_center,
            history_num_frames=history_num_frames,
            history_step_size=history_step_size,
            future_num_frames=future_num_frames,
            future_step_size=future_step_size,
            filter_agents_threshold=self.filter_agents_threshold,
            rasterizer=self.rast,
        )

    def test_no_frames(self) -> None:
        gen_partial = self.get_partial(2, 1, 4, 1)
        with self.assertRaises(IndexError):
            gen_partial(
                state_index=0,
                frames=np.zeros(0, FRAME_DTYPE),
                agents=np.zeros(0, AGENT_DTYPE),
                selected_track_id=None,
            )

    def test_out_bounds(self) -> None:
        gen_partial = self.get_partial(0, 1, 10, 1)
        data = gen_partial(
            state_index=0,
            frames=np.asarray(self.dataset.frames[90:96]),
            agents=self.dataset.agents,
            selected_track_id=None,
        )
        assert bool(np.all(data["target_availabilities"][:5])) is True
        assert bool(np.all(data["target_availabilities"][5:])) is False

    def test_future(self) -> None:
        steps = [(1, 1), (2, 2), (4, 4)]  # all of these should work
        for step, step_size in steps:
            gen_partial = self.get_partial(2, 1, step, step_size)
            data = gen_partial(
                state_index=10,
                frames=np.asarray(self.dataset.frames[90:150]),
                agents=self.dataset.agents,
                selected_track_id=None,
            )
            assert data["target_positions"].shape == (step, 2)
            assert data["target_yaws"].shape == (step, 1)
            assert data["target_availabilities"].shape == (step, 3)
            assert data["centroid"].shape == (2,)
            assert isinstance(data["yaw"], float)
            assert data["extent"].shape == (3,)
            assert bool(np.all(data["target_availabilities"])) is True


if __name__ == "__main__":
    unittest.main()
