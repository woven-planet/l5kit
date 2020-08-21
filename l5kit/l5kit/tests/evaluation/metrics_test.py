import numpy as np
import pytest

from l5kit.evaluation.metrics import (
    _assert_shapes,
    average_displacement_error_mean,
    average_displacement_error_oracle,
    final_displacement_error_mean,
    final_displacement_error_oracle,
    neg_multi_log_likelihood,
    prob_true_mode,
    rmse,
    time_displace,
)


def test_assert_shapes() -> None:
    num_modes, future_len, num_coords = 4, 12, 2

    gt = np.random.randn(future_len, num_coords)
    pred = np.random.randn(num_modes, future_len, num_coords)
    avail = np.ones(future_len)
    conf = np.random.rand(num_modes)
    conf /= np.sum(conf, axis=-1, keepdims=True)

    # test un-normalised conf
    with pytest.raises(AssertionError):
        conf_un = np.random.rand(4)
        _assert_shapes(gt, pred, conf_un, avail)

    # test single pred with no axis
    with pytest.raises(AssertionError):
        _assert_shapes(gt, pred[0], conf, avail)

    # NLL shape must be ()
    assert neg_multi_log_likelihood(gt, pred, conf, avail).shape == ()
    # RMSE shape must be ()
    assert rmse(gt, pred, conf, avail).shape == ()
    # prob_true_mode shape must be (M)
    assert prob_true_mode(gt, pred, conf, avail).shape == (num_modes,)
    # displace_t shape must be (T)
    assert time_displace(gt, pred, conf, avail).shape == (future_len,)


def test_neg_multi_log_likelihood_known_results() -> None:
    # below M=2, T=3, C=1 and CONF=[1,1,1]
    num_modes, future_len, num_coords = 2, 3, 1
    avail = np.ones(future_len)

    gt = np.zeros((future_len, num_coords))
    pred = np.zeros((num_modes, future_len, num_coords))
    pred[0] = [[0], [0], [0]]
    pred[1] = [[10], [10], [10]]

    # single mode, one 100% right
    confs = np.asarray((1, 0))

    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 0)
    assert np.allclose(rmse(gt, pred, confs, avail), 0)

    # two equal modes, one 100% right
    confs = np.asarray((0.5, 0.5))

    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 0.69314, atol=1e-4)
    assert np.allclose(rmse(gt, pred, confs, avail), np.sqrt(2 * 0.69314 / future_len), atol=1e-4)

    # two equal modes, answer in between
    gt = np.full((future_len, num_coords), 5)
    confs = np.asarray((0.5, 0.5))

    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 37.5, atol=1e-4)
    assert np.allclose(rmse(gt, pred, confs, avail), np.sqrt(2 * 37.5 / future_len), atol=1e-4)

    # two modes, one 50% right = answer in between
    gt = np.full((future_len, num_coords), 5)
    confs = np.asarray((1, 0))

    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 37.5, atol=1e-4)
    assert np.allclose(rmse(gt, pred, confs, avail), np.sqrt(2 * 37.5 / future_len), atol=1e-4)

    # Example 5
    gt = np.zeros((future_len, num_coords))
    gt[1, 0] = 10
    confs = np.asarray((1, 0))
    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 50, atol=1e-4)
    assert np.allclose(rmse(gt, pred, confs, avail), np.sqrt(2 * 50 / future_len), atol=1e-4)

    # Example 6
    gt = np.zeros((future_len, num_coords))
    gt[1, 0] = 10
    confs = np.asarray((0.5, 0.5))
    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 50.6931, atol=1e-4)
    assert np.allclose(rmse(gt, pred, confs, avail), np.sqrt(2 * 50.6931 / future_len), atol=1e-4)

    # Test overflow resistance in two situations
    confs = np.asarray((0.5, 0.5))
    pred[0] = [[1000], [1000], [1000]]
    pred[1] = [[1000], [1000], [1000]]
    gt = np.zeros((future_len, num_coords))
    assert not np.isinf(neg_multi_log_likelihood(gt, pred, confs, avail))
    assert not np.isinf(rmse(gt, pred, confs, avail))

    # this breaks also max-version if confidence is not included in exp
    confs = np.asarray((1.0, 0.0))
    pred[0] = [[100000], [1000], [1000]]
    pred[1] = [[1000], [1000], [1000]]
    gt = np.zeros((future_len, num_coords))
    assert not np.isinf(neg_multi_log_likelihood(gt, pred, confs, avail))
    assert not np.isinf(rmse(gt, pred, confs, avail))


def test_other_metrics_known_results() -> None:
    gt = np.asarray([[50, 0], [50, 0], [50, 0]])
    avail = np.ones(3)

    pred = np.asarray([[[50, 0], [50, 0], [50, 0]], [[100, 100], [100, 100], [100, 100]]])
    confs = np.asarray((0.5, 0.5))
    assert np.allclose(prob_true_mode(gt, pred, confs, avail), (1.0, 0.0))
    assert np.allclose(time_displace(gt, pred, confs, avail), (0.0, 0.0, 0.0))

    pred = np.asarray([[[52, 0], [52, 0], [52, 0]], [[49, 0], [51, 0], [50, 2]]])
    confs = np.asarray((0.1, 0.9))
    assert np.allclose(prob_true_mode(gt, pred, confs, avail), (0.0055, 0.9944), atol=1e-4)
    assert np.allclose(time_displace(gt, pred, confs, avail), (1.0055, 1.0055, 2.0), atol=1e-4)


def test_ade_fde_known_results() -> None:
    # below M=2, T=3, C=1 and CONF=[1,1,1]
    num_modes, future_len, num_coords = 2, 3, 1
    avail = np.ones(future_len)

    gt = np.zeros((future_len, num_coords))
    pred = np.zeros((num_modes, future_len, num_coords))
    pred[0] = [[0], [0], [0]]
    pred[1] = [[1], [2], [3]]

    # Confidences do not matter here.
    confs = np.asarray((0, 0))

    assert np.allclose(average_displacement_error_mean(gt, pred, confs, avail), 1)
    assert np.allclose(average_displacement_error_oracle(gt, pred, confs, avail), 0)
    assert np.allclose(final_displacement_error_mean(gt, pred, confs, avail), 1.5)
    assert np.allclose(final_displacement_error_oracle(gt, pred, confs, avail), 0)

    gt = np.full((future_len, num_coords), 0.5)

    assert np.allclose(average_displacement_error_mean(gt, pred, confs, avail), 1.0)
    assert np.allclose(average_displacement_error_oracle(gt, pred, confs, avail), 0.5)
    assert np.allclose(final_displacement_error_mean(gt, pred, confs, avail), 1.5)
    assert np.allclose(final_displacement_error_oracle(gt, pred, confs, avail), 0.5)
