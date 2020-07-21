from pathlib import Path

import numpy as np

from l5kit.visualization import write_video


def test_write_video(tmp_path: Path) -> None:
    # Just a smoke test
    images = (np.random.rand(5, 512, 512, 3) * 255).astype(np.uint8)

    video_filepath = tmp_path / "test_video.mp4"
    write_video(str(video_filepath), images, (512, 512))

    assert video_filepath.exists()


def test_write_video_with_resize(tmp_path: Path) -> None:
    # Just a smoke test
    images = (np.random.rand(5, 256, 256, 3) * 255).astype(np.uint8)

    video_filepath = tmp_path / "test_video.mp4"
    write_video(str(video_filepath), images, (512, 512))

    assert video_filepath.exists()
    assert video_filepath.is_file()


def test_write_video_bw(tmp_path: Path) -> None:
    # Just a smoke test
    images = (np.random.rand(5, 512, 512) * 255).astype(np.uint8)

    video_filepath = tmp_path / "test_video.mp4"
    write_video(str(video_filepath), images, (512, 512))

    assert video_filepath.exists()
    assert video_filepath.is_file()
