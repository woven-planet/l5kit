import unittest

import numpy as np

from l5kit.data import AGENT_DTYPE, ChunkedStateDataset, get_frames_agents
from l5kit.rasterization.box_rasterizer import BoxRasterizer, draw_boxes


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


class BoxRasterizerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):  # type: ignore
        super(BoxRasterizerTest, self).__init__(*args, **kwargs)
        self.dataset = ChunkedStateDataset(path="./l5kit/tests/data/single_scene.zarr")
        self.dataset.open()
        self.hist_frames = self.dataset.frames[100:101]  # we know this has agents
        self.hist_agents = get_frames_agents(self.hist_frames, self.dataset.agents)

    def test_ego_center(self) -> None:
        values = [(0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75)]
        for v in values:
            rast = BoxRasterizer(
                (224, 224),
                np.asarray((0.25, 0.25)),
                ego_center=np.asarray(v),
                filter_agents_threshold=0.0,
                history_num_frames=0,
            )
            out = rast.rasterize(self.hist_frames, self.hist_agents)
            assert out[..., -1].sum() > 0

    def test_agents_map(self) -> None:
        rast = BoxRasterizer((224, 224), np.asarray((0.25, 0.25)), np.asarray((0.25, 0.5)), 1.0, 0)
        out = rast.rasterize(self.hist_frames, self.hist_agents)
        assert out[..., 0].sum() == 0

        rast = BoxRasterizer((224, 224), np.asarray((0.25, 0.25)), np.asarray((0.25, 0.5)), 0.0, 0)
        out = rast.rasterize(self.hist_frames, self.hist_agents)
        assert out[..., 0].sum() > 0

    def test_agent_ego(self) -> None:
        rast = BoxRasterizer((224, 224), np.asarray((0.25, 0.25)), np.asarray((0.25, 0.5)), -1, 0)

        agents = self.hist_agents[0]
        for ag in agents:
            out = rast.rasterize(self.hist_frames, self.hist_agents, ag)
            assert out[..., -1].sum() > 0

    def test_shape(self) -> None:
        hist_length = 10
        rast = BoxRasterizer(
            (224, 224), np.asarray((0.25, 0.25)), np.asarray((0.25, 0.5)), -1, history_num_frames=hist_length
        )
        out = rast.rasterize(self.dataset.frames[: hist_length + 1], self.hist_agents[: hist_length + 1])
        assert out.shape == (224, 224, (hist_length + 1) * 2)
