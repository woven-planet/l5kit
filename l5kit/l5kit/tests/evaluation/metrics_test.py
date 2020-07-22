import numpy as np
import pytest

from l5kit.evaluation.metrics import multi_trajectory_metrics, single_trajectory_metrics


def test_single_trajectory_metrics_shapes() -> None:
    gt = np.random.randn(10, 12, 2)
    pred = np.random.randn(10, 12, 2)
    assert single_trajectory_metrics(gt, pred).shape == (12,)

    # remove batch dim
    with pytest.raises(AssertionError):
        single_trajectory_metrics(gt[0], pred)

    # different steps
    with pytest.raises(AssertionError):
        single_trajectory_metrics(gt, pred[:, :10])

    # different coords
    with pytest.raises(AssertionError):
        single_trajectory_metrics(gt, pred[:, :, :1])


def test_single_trajectory_metrics() -> None:
    assert np.allclose(single_trajectory_metrics(np.zeros((10, 12, 2)), np.zeros((10, 12, 2))), 0)

    # test single trajectory with known error
    gt = np.zeros((1, 3, 2))
    pred = np.zeros((1, 3, 2))
    pred[0] = [[0, 0], [10, 10], [0, 0]]

    assert np.allclose(single_trajectory_metrics(gt, pred), [0, 14.1421, 0])


def test_multi_trajectory_metrics_shapes() -> None:
    gt = np.random.randn(10, 12, 2)
    pred = np.random.randn(10, 4, 12, 2)
    confs = np.random.rand(10, 4)
    confs /= np.sum(confs, axis=-1, keepdims=True)

    assert multi_trajectory_metrics(gt, pred, confs).shape == ()

    # add multiple gt
    with pytest.raises(AssertionError):
        multi_trajectory_metrics(pred, pred, confs)


def test_multi_trajectory_metrics_confidences() -> None:
    gt = np.random.randn(10, 12, 2)
    pred = np.random.randn(10, 4, 12, 2)

    # generate un-normalised confs
    confs = np.random.rand(10, 4)
    confs[0, 3] = 10  # ensures assertion will be triggered

    with pytest.raises(AssertionError):
        multi_trajectory_metrics(gt, pred, confs)


# TODO add metrics test with known values
