from pathlib import Path

import numpy as np
import pytest

from l5kit.data import ChunkedDataset
from l5kit.evaluation import compute_mse_error_csv, export_zarr_to_ground_truth_csv


def test_compute_mse_error(tmp_path: Path, zarr_dataset: ChunkedDataset) -> None:
    export_zarr_to_ground_truth_csv(zarr_dataset, str(tmp_path / "gt1.csv"), 10, 50, 0.5)
    export_zarr_to_ground_truth_csv(zarr_dataset, str(tmp_path / "gt2.csv"), 10, 50, 0.5)
    err = compute_mse_error_csv(str(tmp_path / "gt1.csv"), str(tmp_path / "gt2.csv"))
    assert np.all(err == 0.0)

    data_fake = ChunkedDataset(str(tmp_path))
    data_fake.scenes = np.asarray(zarr_dataset.scenes).copy()
    data_fake.frames = np.asarray(zarr_dataset.frames).copy()
    data_fake.agents = np.asarray(zarr_dataset.agents).copy()
    data_fake.agents["centroid"] += np.random.rand(*data_fake.agents["centroid"].shape) * 1e-2

    export_zarr_to_ground_truth_csv(data_fake, str(tmp_path / "gt3.csv"), 10, 50, 0.5)
    err = compute_mse_error_csv(str(tmp_path / "gt1.csv"), str(tmp_path / "gt3.csv"))
    assert np.any(err > 0.0)

    # test invalid conf by removing lines in gt1
    with open(str(tmp_path / "gt4.csv"), "w") as fp:
        lines = open(str(tmp_path / "gt1.csv")).readlines()
        fp.writelines(lines[:-10])

    with pytest.raises(ValueError):
        compute_mse_error_csv(str(tmp_path / "gt1.csv"), str(tmp_path / "gt4.csv"))
