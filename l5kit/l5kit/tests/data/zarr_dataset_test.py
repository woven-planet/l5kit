from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.zarr_utils import zarr_concat


def test_load_dataset(zarr_dataset: ChunkedDataset) -> None:
    assert len(zarr_dataset.frames) > 0
    assert len(zarr_dataset.agents) > 0
    assert len(zarr_dataset.scenes) > 0
    assert len(zarr_dataset.tl_faces) > 0

    # check first and last frames timestamps differ
    assert zarr_dataset.frames[0]["timestamp"] != zarr_dataset.frames[-1]["timestamp"]

    # check positions differ
    assert np.any(zarr_dataset.frames[0]["ego_translation"] != zarr_dataset.frames[1]["ego_translation"])


def test_get_scene_dataset(dmg: LocalDataManager, tmp_path: Path, zarr_dataset: ChunkedDataset) -> None:
    concat_count = 4
    zarr_input_path = dmg.require("single_scene.zarr")
    zarr_output_path = str(tmp_path / f"{uuid4()}.zarr")

    zarr_concat([zarr_input_path] * concat_count, zarr_output_path)
    zarr_cat_dataset = ChunkedDataset(zarr_output_path)
    zarr_cat_dataset.open()

    # all scenes should be the same as the input one
    for scene_idx in range(concat_count):
        zarr_scene = zarr_cat_dataset.get_scene_dataset(scene_idx)
        assert np.alltrue(zarr_scene.scenes == np.asarray(zarr_dataset.scenes))
        assert np.alltrue(zarr_scene.frames == np.asarray(zarr_dataset.frames))
        assert np.alltrue(zarr_scene.agents == np.asarray(zarr_dataset.agents))
        assert np.alltrue(zarr_scene.tl_faces == np.asarray(zarr_dataset.tl_faces))

    with pytest.raises(ValueError):
        zarr_cat_dataset.get_scene_dataset(concat_count + 1)
