from pathlib import Path

import numpy as np
import pytest

from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.evaluation import compute_metrics_csv, write_gt_csv, write_pred_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood
from l5kit.rasterization import RenderContext, StubRasterizer


def test_compute_mse_error(tmp_path: Path, zarr_dataset: ChunkedDataset, cfg: dict) -> None:
    render_context = RenderContext(
        np.asarray((10, 10)),
        np.asarray((0.25, 0.25)),
        np.asarray((0.5, 0.5)),
        set_origin_to_bottom=cfg["raster_params"]["set_origin_to_bottom"],
    )
    rast = StubRasterizer(render_context)
    dataset = AgentDataset(cfg, zarr_dataset, rast)

    gt_coords = []
    gt_avails = []
    timestamps = []
    track_ids = []

    for idx, el in enumerate(dataset):  # type: ignore
        gt_coords.append(el["target_positions"])
        gt_avails.append(el["target_availabilities"])
        timestamps.append(el["timestamp"])
        track_ids.append(el["track_id"])
        if idx == 100:
            break  # speed up test

    gt_coords = np.asarray(gt_coords)
    gt_avails = np.asarray(gt_avails)
    timestamps = np.asarray(timestamps)
    track_ids = np.asarray(track_ids)

    # test same values error
    write_gt_csv(str(tmp_path / "gt1.csv"), timestamps, track_ids, gt_coords, gt_avails)
    write_pred_csv(str(tmp_path / "pred1.csv"), timestamps, track_ids, gt_coords, confs=None)

    metrics = compute_metrics_csv(str(tmp_path / "gt1.csv"), str(tmp_path / "pred1.csv"), [neg_multi_log_likelihood])
    for metric_value in metrics.values():
        assert np.all(metric_value == 0.0)

    # test different values error
    pred_coords = gt_coords.copy()
    pred_coords += np.random.randn(*pred_coords.shape)
    write_pred_csv(str(tmp_path / "pred3.csv"), timestamps, track_ids, pred_coords, confs=None)

    metrics = compute_metrics_csv(str(tmp_path / "gt1.csv"), str(tmp_path / "pred3.csv"), [neg_multi_log_likelihood])
    for metric_value in metrics.values():
        assert np.any(metric_value > 0.0)

    # test invalid conf by removing lines in gt1
    with open(str(tmp_path / "pred4.csv"), "w") as fp:
        lines = open(str(tmp_path / "pred1.csv")).readlines()
        fp.writelines(lines[:-10])

    with pytest.raises(ValueError):
        compute_metrics_csv(str(tmp_path / "gt1.csv"), str(tmp_path / "pred4.csv"), [neg_multi_log_likelihood])
