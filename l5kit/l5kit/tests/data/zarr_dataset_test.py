import numpy as np

from l5kit.data import ChunkedDataset


def test_load_dataset(zarr_dataset: ChunkedDataset) -> None:
    assert len(zarr_dataset.frames) > 0
    assert len(zarr_dataset.agents) > 0
    assert len(zarr_dataset.scenes) > 0
    assert len(zarr_dataset.tl_faces) > 0

    # check first and last frames timestamps differ
    assert zarr_dataset.frames[0]["timestamp"] != zarr_dataset.frames[-1]["timestamp"]

    # check positions differ
    assert np.any(zarr_dataset.frames[0]["ego_translation"] != zarr_dataset.frames[1]["ego_translation"])
