from pathlib import Path

from l5kit.data import zarr_dataset
from l5kit.evaluation import extract_ground_truth


def test_export_to_zarr(tmp_path: Path) -> None:

    data = zarr_dataset.ChunkedStateDataset(path="./l5kit/tests/artefacts/single_scene.zarr")
    data.open()
    tmp_path = tmp_path / "out.csv"
    extract_ground_truth.export_zarr_to_ground_truth_csv(
        data, str(tmp_path), history_num_frames=0, future_num_frames=12, filter_agents_threshold=0.5
    )
    assert tmp_path.exists()
