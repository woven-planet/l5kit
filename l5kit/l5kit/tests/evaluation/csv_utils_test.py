from pathlib import Path

import numpy as np
import pytest

from l5kit.evaluation import read_gt_csv, read_pred_csv, write_gt_csv, write_pred_csv


def test_write_gt_csv(tmpdir: Path) -> None:
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


def test_e2e_gt_csv(tmpdir: Path) -> None:
    dump_path = str(tmpdir / "gt_out.csv")
    num_example, future_len, num_coords = 100, 12, 2

    timestamps = np.random.randint(1000, 2000, num_example)
    track_ids = np.random.randint(0, 200, num_example)
    coords = np.random.randn(*(num_example, future_len, num_coords))
    avails = np.random.randint(0, 2, (num_example, future_len))
    write_gt_csv(dump_path, timestamps, track_ids, coords, avails)

    # read and check values
    for idx, el in enumerate(read_gt_csv(dump_path)):
        assert int(el["track_id"]) == track_ids[idx]
        assert int(el["timestamp"]) == timestamps[idx]
        assert np.allclose(el["coord"], coords[idx], atol=1e-4)
        assert np.allclose(el["avail"], avails[idx])


def test_write_pred_csv(tmpdir: Path) -> None:
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
    confs /= np.sum(confs, axis=1, keepdims=True)

    dump_path = str(tmpdir / "pred_pred_multi.csv")
    write_pred_csv(dump_path, timestamps, track_ids, coords, confs)
    assert Path(dump_path).exists()


def test_e2e_multi_pred_csv(tmpdir: Path) -> None:
    dump_path = str(tmpdir / "pred_out.csv")
    num_example, num_modes, future_len, num_coords = 100, 3, 12, 2

    timestamps = np.random.randint(1000, 2000, num_example)
    track_ids = np.random.randint(0, 200, num_example)

    coords = np.random.randn(*(num_example, num_modes, future_len, num_coords))
    confs = np.random.rand(*(num_example, num_modes))
    confs /= np.sum(confs, axis=1, keepdims=True)

    write_pred_csv(dump_path, timestamps, track_ids, coords, confs)

    # read and check values
    for idx, el in enumerate(read_pred_csv(dump_path)):
        assert int(el["track_id"]) == track_ids[idx]
        assert int(el["timestamp"]) == timestamps[idx]
        assert np.allclose(el["coords"], coords[idx], atol=1e-4)
        assert np.allclose(el["conf"], confs[idx], atol=1e-4)


def test_e2e_single_pred_csv(tmpdir: Path) -> None:
    dump_path = str(tmpdir / "pred_out.csv")
    num_example, future_len, num_coords = 100, 12, 2

    timestamps = np.random.randint(1000, 2000, num_example)
    track_ids = np.random.randint(0, 200, num_example)

    coords = np.random.randn(*(num_example, future_len, num_coords))
    write_pred_csv(dump_path, timestamps, track_ids, coords, confs=None)

    # read and check values
    for idx, el in enumerate(read_pred_csv(dump_path)):
        assert int(el["track_id"]) == track_ids[idx]
        assert int(el["timestamp"]) == timestamps[idx]
        assert np.allclose(el["coords"][0], coords[idx], atol=1e-4)
        assert el["conf"][0] == 1

        assert np.allclose(el["coords"][1:], 0)
        assert np.allclose(el["conf"][1:], 0)
