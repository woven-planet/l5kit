import numpy as np
import pytest

from l5kit.evaluation.metrics import multi_trajectory_metric, single_trajectory_metric


def test_single_trajectory_metrics_shapes() -> None:
    gt = np.random.randn(10, 12, 2)
    pred = np.random.randn(10, 12, 2)
    assert single_trajectory_metric(gt, pred).shape == (12,)

    # remove batch dim
    with pytest.raises(AssertionError):
        single_trajectory_metric(gt[0], pred)

    # different steps
    with pytest.raises(AssertionError):
        single_trajectory_metric(gt, pred[:, :10])

    # different coords
    with pytest.raises(AssertionError):
        single_trajectory_metric(gt, pred[:, :, :1])


def test_single_trajectory_metrics() -> None:
    # 0 error
    assert np.allclose(single_trajectory_metric(np.zeros((10, 12, 2)), np.zeros((10, 12, 2))), 0)

    # test single trajectory with known error
    gt = np.zeros((1, 3, 2))
    pred = np.zeros((1, 3, 2))
    pred[0] = [[0, 0], [10, 10], [0, 0]]

    assert np.allclose(single_trajectory_metric(gt, pred), [0, 14.1421, 0], atol=1e-3)


def test_multi_trajectory_metrics_shapes() -> None:
    gt = np.random.randn(10, 12, 2)
    pred = np.random.randn(10, 4, 12, 2)
    confs = np.random.rand(10, 4)
    confs /= np.sum(confs, axis=-1, keepdims=True)

    assert multi_trajectory_metric(gt, pred, confs).shape == (12,)

    # add multiple gt
    with pytest.raises(AssertionError):
        multi_trajectory_metric(pred, pred, confs)


def test_multi_trajectory_metrics_confidences() -> None:
    gt = np.random.randn(10, 12, 2)
    pred = np.random.randn(10, 4, 12, 2)

    # generate un-normalised confs
    confs = np.random.rand(10, 4)
    confs[0, 3] = 10  # ensures assertion will be triggered

    with pytest.raises(AssertionError):
        multi_trajectory_metric(gt, pred, confs)


def test_multi_trajectory_metrics() -> None:
    # below M=2, T=3, C=1

    # 100% on correct one
    gt = np.zeros((1, 3, 1))
    gt[0] = [[0], [0], [0]]

    pred = np.zeros((1, 2, 3, 1))
    pred[0, 0] = [[0], [0], [0]]
    pred[0, 1] = [[10], [10], [10]]

    confs = np.zeros((1, 2))
    confs[0] = [1, 0]

    assert np.allclose(multi_trajectory_metric(gt, pred, confs), [0, 0, 0])

    # 50% on correct one
    confs[0] = [0.5, 0.5]
    assert np.allclose(multi_trajectory_metric(gt, pred, confs), [0.69314, 0.69314, 0.69314], atol=1e-4)

    # answer in between, conf 0.5
    gt[0] = [[5], [5], [5]]
    confs[0] = [0.5, 0.5]
    assert np.allclose(multi_trajectory_metric(gt, pred, confs), [12.5, 12.5, 12.5], atol=1e-4)

    # example 5
    gt[0] = [[0], [10], [0]]
    confs[0] = [1, 0]
    assert np.allclose(multi_trajectory_metric(gt, pred, confs), [0, 50, 0], atol=1e-4)

    # example 6
    gt[0] = [[0], [10], [0]]
    confs[0] = [0.5, 0.5]
    assert np.allclose(multi_trajectory_metric(gt, pred, confs), [0.69314, 0.69314, 0.69314], atol=1e-4)
