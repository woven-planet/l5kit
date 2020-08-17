import numpy as np
import pytest

from l5kit.evaluation.metrics import neg_multi_log_likelihood, prob_true_mode, time_displace


def test_neg_multi_log_likelihood_shapes() -> None:
    gt = np.random.randn(12, 2)
    pred = np.random.randn(4, 12, 2)
    avail = np.ones(12)
    conf = np.random.rand(4)
    conf /= np.sum(conf, axis=-1, keepdims=True)

    assert neg_multi_log_likelihood(gt, pred, conf, avail).shape == ()

    # test unnormalised conf
    conf = np.random.rand(4)
    with pytest.raises(AssertionError):
        neg_multi_log_likelihood(gt, pred, conf, avail)

    # test single pred with no axis
    with pytest.raises(AssertionError):
        neg_multi_log_likelihood(gt, pred[0], conf, avail)


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

    # two equal modes, one 100% right
    confs = np.asarray((0.5, 0.5))

    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 0.69314, atol=1e-4)

    # two equal modes, answer in between
    gt = np.full((future_len, num_coords), 5)
    confs = np.asarray((0.5, 0.5))

    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 37.5, atol=1e-4)

    # two modes, one 50% right = answer in between
    gt = np.full((future_len, num_coords), 5)
    confs = np.asarray((1, 0))

    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 37.5, atol=1e-4)

    # Example 5
    gt = np.zeros((future_len, num_coords))
    gt[1, 0] = 10
    confs = np.asarray((1, 0))
    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 50, atol=1e-4)

    # Example 6
    gt = np.zeros((future_len, num_coords))
    gt[1, 0] = 10
    confs = np.asarray((0.5, 0.5))
    assert np.allclose(neg_multi_log_likelihood(gt, pred, confs, avail), 50.6931, atol=1e-4)

    # Test overflow resistance in two situations
    confs = np.asarray((0.5, 0.5))
    pred[0] = [[1000], [1000], [1000]]
    pred[1] = [[1000], [1000], [1000]]
    gt = np.zeros((future_len, num_coords))
    assert not np.isinf(neg_multi_log_likelihood(gt, pred, confs, avail))

    # this breaks also max-version if confidence is not included in exp
    confs = np.asarray((1.0, 0.0))
    pred[0] = [[100000], [1000], [1000]]
    pred[1] = [[1000], [1000], [1000]]
    gt = np.zeros((future_len, num_coords))
    assert not np.isinf(neg_multi_log_likelihood(gt, pred, confs, avail))
