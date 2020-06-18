import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from l5kit.visualization import write_gif


class TestGifVisualizationHelpers(unittest.TestCase):
    def test_write_gif(self) -> None:
        # Just a smoke test
        images = (np.random.rand(5, 512, 512, 3) * 255).astype(np.uint8)

        with TemporaryDirectory() as d:
            gif_filepath = os.path.join(d, "test_gif.gif")
            write_gif(gif_filepath, images, (512, 512))

            self.assertTrue(os.path.isfile(gif_filepath))

    def test_write_gif_with_resize(self) -> None:
        # Just a smoke test
        images = (np.random.rand(5, 512, 512, 3) * 255).astype(np.uint8)

        with TemporaryDirectory() as d:
            gif_filepath = os.path.join(d, "test_gif.gif")
            write_gif(gif_filepath, images, (256, 256))

            self.assertTrue(os.path.isfile(gif_filepath))

    def test_write_gif_bw(self) -> None:
        # Just a smoke test
        images = (np.random.rand(5, 512, 512) * 255).astype(np.uint8)

        with TemporaryDirectory() as d:
            gif_filepath = os.path.join(d, "test_gif.gif")
            write_gif(gif_filepath, images, (512, 512))

            self.assertTrue(os.path.isfile(gif_filepath))
