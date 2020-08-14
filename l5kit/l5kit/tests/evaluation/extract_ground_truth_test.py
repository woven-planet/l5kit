from pathlib import Path

from l5kit.data import ChunkedDataset
from l5kit.evaluation import extract_ground_truth


def test_export_to_zarr(tmp_path: Path, zarr_dataset: ChunkedDataset) -> None:
    tmp_path = tmp_path / "out.csv"
    extract_ground_truth.export_zarr_to_csv(
        zarr_dataset, str(tmp_path), future_num_frames=12, filter_agents_threshold=0.5
    )
    assert tmp_path.exists()
