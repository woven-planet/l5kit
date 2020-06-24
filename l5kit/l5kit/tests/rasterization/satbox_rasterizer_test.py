import unittest

import numpy as np

from l5kit.data import ChunkedStateDataset
from l5kit.rasterization import SatBoxRasterizer


class SatBoxRasterizerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):  # type: ignore
        super(SatBoxRasterizerTest, self).__init__(*args, **kwargs)
        self.dataset = ChunkedStateDataset(path="./l5kit/tests/data/single_scene.zarr")
        self.dataset.open()

    def test_shape(self) -> None:
        map_to_sat = np.block(
            [[np.eye(3) / 100, np.asarray([[1000], [1000], [1]])], [np.asarray([[0, 0, 0, 1]])]]
        )  # just a translation and scale
        hist_length = 10

        rast = SatBoxRasterizer(
            (224, 224),
            np.asarray((0.25, 0.25)),
            np.asarray((0.25, 0.5)),
            filter_agents_threshold=-1,
            history_num_frames=hist_length,
            map_im=np.zeros((10000, 10000, 3), dtype=np.uint8),
            map_to_sat=map_to_sat,
        )

        out = rast.rasterize(self.dataset.frames[: hist_length + 1], self.dataset.agents)
        assert out.shape == (224, 224, (hist_length + 1) * 2 + 3)
