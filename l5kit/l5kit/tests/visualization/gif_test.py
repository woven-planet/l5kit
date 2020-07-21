from pathlib import Path

import numpy as np

from l5kit.visualization import write_gif


def test_write_gif(tmp_path: Path) -> None:
    # Just a smoke test
    images = (np.random.rand(5, 512, 512, 3) * 255).astype(np.uint8)

    video_filepath = tmp_path / "test_gif.gif"
    write_gif(str(video_filepath), images, (512, 512))

    assert video_filepath.exists()


def test_write_gif_with_resize(tmp_path: Path) -> None:
    # Just a smoke test
    images = (np.random.rand(5, 256, 256, 3) * 255).astype(np.uint8)

    video_filepath = tmp_path / "test_gif.gif"
    write_gif(str(video_filepath), images, (512, 512))

    assert video_filepath.exists()
    assert video_filepath.is_file()


def test_write_gif_bw(tmp_path: Path) -> None:
    # Just a smoke test
    images = (np.random.rand(5, 512, 512) * 255).astype(np.uint8)

    video_filepath = tmp_path / "test_gif.gif"
    write_gif(str(video_filepath), images, (512, 512))

    assert video_filepath.exists()
    assert video_filepath.is_file()
