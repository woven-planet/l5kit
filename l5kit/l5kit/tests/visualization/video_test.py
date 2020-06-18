import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from l5kit.visualization import write_video


class TestVideoVisualizationHelpers(unittest.TestCase):
    def test_write_video(self) -> None:
        # Just a smoke test
        images = (np.random.rand(5, 512, 512, 3) * 255).astype(np.uint8)

        with TemporaryDirectory() as d:
            video_filepath = os.path.join(d, "test_video.mp4")
            write_video(video_filepath, images, (512, 512))

            self.assertTrue(os.path.isfile(video_filepath))

    def test_write_video_with_resize(self) -> None:
        # Just a smoke test
        images = (np.random.rand(5, 256, 256, 3) * 255).astype(np.uint8)

        with TemporaryDirectory() as d:
            video_filepath = os.path.join(d, "test_video.mp4")
            write_video(video_filepath, images, (512, 512))

            self.assertTrue(os.path.isfile(video_filepath))

    def test_write_video_bw(self) -> None:
        # Just a smoke test
        images = (np.random.rand(5, 512, 512) * 255).astype(np.uint8)

        with TemporaryDirectory() as d:
            video_filepath = os.path.join(d, "test_video.mp4")
            write_video(video_filepath, images, (512, 512))

            self.assertTrue(os.path.isfile(video_filepath))
