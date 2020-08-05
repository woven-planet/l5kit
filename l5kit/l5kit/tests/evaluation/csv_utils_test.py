from pathlib import Path

import numpy as np
import pytest

from l5kit.evaluation import read_gt_csv, read_pred_csv, write_gt_csv, write_pred_csv


def test_write_gt_csv(tmpdir: Path):
    dump_path = str(tmpdir / "gt_out.csv")
    num_example, future_len, num_coords = 100, 12, 2

    timestamps = np.zeros(num_example)
    track_ids = np.zeros(num_example)

    # test some invalid shapes for coords and avails
    with pytest.raises(AssertionError):
        coords = np.zeros((num_example, 2, future_len, num_coords))  # gt multi-modal
        avails = np.zeros((num_example, future_len))
        write_gt_csv(dump_path, timestamps, track_ids, coords, avails)
    with pytest.raises(AssertionError):
        coords = np.zeros((num_example, future_len, num_coords))
        avails = np.zeros((num_example, future_len + 5))  # mismatch
        write_gt_csv(dump_path, timestamps, track_ids, coords, avails)
    with pytest.raises(AssertionError):
        coords = np.zeros((num_example, future_len, num_coords))
        avails = np.zeros((num_example, future_len, num_coords))  # avails per coords
        write_gt_csv(dump_path, timestamps, track_ids, coords, avails)

    # test a valid configuration
    coords = np.zeros((num_example, future_len, num_coords))
    avails = np.zeros((num_example, future_len))
    write_gt_csv(dump_path, timestamps, track_ids, coords, avails)
    assert Path(dump_path).exists()


def test_write_pred_csv(tmpdir: Path):
    dump_path = str(tmpdir / "pred_pred.csv")
    num_example, num_modes, future_len, num_coords = 100, 3, 12, 2

    timestamps = np.zeros(num_example)
    track_ids = np.zeros(num_example)

    # test some invalid shapes for coords and confidences
    with pytest.raises(AssertionError):
        coords = np.zeros((num_example, future_len, num_coords))  # pred with 1 mode and confidence 1
        confs = np.ones((num_example,))
        write_pred_csv(dump_path, timestamps, track_ids, coords, confs)
    with pytest.raises(AssertionError):
        coords = np.zeros((num_example, num_modes, future_len, num_coords))
        confs = np.ones((num_example, 1))  # no modes
        write_pred_csv(dump_path, timestamps, track_ids, coords, confs)

    # test a valid single-mode configuration
    coords = np.zeros((num_example, future_len, num_coords))
    confs = None
    dump_path = str(tmpdir / "pred_pred_uni.csv")
    write_pred_csv(dump_path, timestamps, track_ids, coords, confs)
    assert Path(dump_path).exists()

    # test a valid multi-mode configuration
    coords = np.zeros((num_example, num_modes, future_len, num_coords))
    confs = np.ones((num_example, num_modes))
    dump_path = str(tmpdir / "pred_pred_multi.csv")
    write_pred_csv(dump_path, timestamps, track_ids, coords, confs)
    assert Path(dump_path).exists()
