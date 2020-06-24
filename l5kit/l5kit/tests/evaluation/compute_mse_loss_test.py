from pathlib import Path

import numpy as np
import pytest

from l5kit.data import ChunkedStateDataset
from l5kit.evaluation import compute_mse_error_csv, export_zarr_to_ground_truth_csv


def test_compute_mse_error(tmp_path: Path) -> None:
    data = ChunkedStateDataset(path="./l5kit/tests/data/single_scene.zarr")
    data.open()
    export_zarr_to_ground_truth_csv(data, str(tmp_path / "gt1.csv"), 0, 12, 0.5)
    export_zarr_to_ground_truth_csv(data, str(tmp_path / "gt2.csv"), 0, 12, 0.5)
    err = compute_mse_error_csv(str(tmp_path / "gt1.csv"), str(tmp_path / "gt2.csv"))
    assert np.all(err == 0.0)

    data_fake = ChunkedStateDataset("")
    data_fake.scenes = np.asarray(data.scenes).copy()
    data_fake.frames = np.asarray(data.frames).copy()
    data_fake.agents = np.asarray(data.agents).copy()
    data_fake.root = data.root
    data_fake.agents["centroid"] += np.random.rand(*data_fake.agents["centroid"].shape)

    export_zarr_to_ground_truth_csv(data_fake, str(tmp_path / "gt3.csv"), 0, 12, 0.5)
    err = compute_mse_error_csv(str(tmp_path / "gt1.csv"), str(tmp_path / "gt3.csv"))
    assert np.any(err > 0.0)

    # test invalid conf by removing lines in gt1
    with open(str(tmp_path / "gt4.csv"), "w") as fp:
        lines = open(str(tmp_path / "gt1.csv")).readlines()
        fp.writelines(lines[:-10])

    with pytest.raises(ValueError):
        compute_mse_error_csv(str(tmp_path / "gt1.csv"), str(tmp_path / "gt4.csv"))
