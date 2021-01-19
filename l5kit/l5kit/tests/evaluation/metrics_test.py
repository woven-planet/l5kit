import numpy as np
import pytest
import torch

from l5kit.evaluation.metrics import (_assert_shapes, average_displacement_error_mean,
                                      average_displacement_error_oracle, CollisionType, detect_collision,
                                      distance_to_reference_trajectory, final_displacement_error_mean,
                                      final_displacement_error_oracle, neg_multi_log_likelihood, prob_true_mode, rmse,
                                      time_displace)
from l5kit.planning.utils import _get_bounding_box, _get_sides, within_range


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
    confs = np.asarray((0.5, 0.5))

    assert np.allclose(average_displacement_error_mean(gt, pred, confs, avail), 1)
    assert np.allclose(average_displacement_error_oracle(gt, pred, confs, avail), 0)
    assert np.allclose(final_displacement_error_mean(gt, pred, confs, avail), 1.5)
    assert np.allclose(final_displacement_error_oracle(gt, pred, confs, avail), 0)

    gt = np.full((future_len, num_coords), 0.5)

    assert np.allclose(average_displacement_error_mean(gt, pred, confs, avail), 1.0)
    assert np.allclose(average_displacement_error_oracle(gt, pred, confs, avail), 0.5)
    assert np.allclose(final_displacement_error_mean(gt, pred, confs, avail), 1.5)
    assert np.allclose(final_displacement_error_oracle(gt, pred, confs, avail), 0.5)


def test_within_range() -> None:
    # Fixture: overlapping ego and agent
    ego_centroid = np.array([[10., 10.]])
    ego_extent = np.array([[5.0, 2.0, 2.0]])
    agent_centroid = np.array([[10.0, 10.0]])
    agent_extent = np.array([[5., 2., 2.]])

    # Ovelarpping ego and agent should be within range
    assert within_range(ego_centroid, ego_extent,
                        agent_centroid, agent_extent)

    # The contrary is also true
    assert within_range(agent_centroid, agent_extent,
                        ego_centroid, ego_extent)

    # Agent is far from the ego, not within range
    assert not within_range(ego_centroid, ego_extent,
                            agent_centroid + 1000.0, agent_extent)

    # Repeat dimension (10, D)
    num_repeat = 10
    ego_centroid = ego_centroid.repeat(num_repeat, axis=0)
    ego_extent = ego_extent.repeat(num_repeat, axis=0)

    agent_centroid = agent_centroid.repeat(num_repeat, axis=0)
    agent_extent = agent_extent.repeat(num_repeat, axis=0)

    truth_value = within_range(ego_centroid, ego_extent,
                               agent_centroid, agent_extent)
    assert len(truth_value) == num_repeat
    assert truth_value.all()

    # Only half not within range
    ego_centroid[5:, :] += 1000.0
    truth_value = within_range(ego_centroid, ego_extent,
                               agent_centroid, agent_extent)
    assert len(truth_value) == num_repeat
    assert np.count_nonzero(truth_value) == 5


def test_get_bounding_box() -> None:
    agent_centroid = np.array([10., 10.])
    agent_extent = np.array([5.0, 2.0, 2.0])
    agent_yaw = 1.0

    bbox = _get_bounding_box(agent_centroid, agent_yaw, agent_extent)

    # Check centroid coordinates and other preoperties of the polygon
    assert np.allclose(bbox.centroid.coords, agent_centroid)
    assert np.allclose(bbox.area, 10.0)
    assert not bbox.is_empty
    assert bbox.is_valid


def test_get_sides() -> None:
    agent_centroid = np.array([10., 10.])
    agent_extent = np.array([5.0, 2.0, 2.0])
    agent_yaw = 1.0

    bbox = _get_bounding_box(agent_centroid, agent_yaw, agent_extent)
    front, rear, left, right = _get_sides(bbox)

    # The parallel offset of s1 should be the same as s2
    coords_parallel_s1 = np.array(front.parallel_offset(agent_extent[0], 'right').coords)
    coords_s2 = np.array(rear.coords)
    assert np.allclose(coords_s2, coords_parallel_s1)

    # One side shouldn't touch the other parallel side
    assert not front.touches(rear)

    # .. but should touch other ortoghonal
    assert front.touches(left)
    assert front.touches(right)

    assert np.allclose(left.length, agent_extent[0])
    assert np.allclose(right.length, agent_extent[0])

    assert np.allclose(front.length, agent_extent[1])
    assert np.allclose(rear.length, agent_extent[1])


def test_detect_collision() -> None:
    pred_centroid = np.array([0.0, 0.0])
    pred_yaw = np.array([1.0])
    pred_extent = np.array([5., 2., 2.])

    target_agents_dtype = np.dtype([('centroid', '<f8', (2,)),
                                    ('extent', '<f4', (3,)),
                                    ('yaw', '<f4'), ('track_id', '<u8')])

    target_agents = np.array([
        ([1000.0, 1000.0], [5., 2., 2.], 1.0, 1),  # Not in range
        ([0., 0.], [5., 2., 2.], 1.0, 2),  # In range
        ([0., 0.], [5., 2., 2.], 1.0, 3),  # In range
    ], dtype=target_agents_dtype)

    collision = detect_collision(pred_centroid, pred_yaw, pred_extent, target_agents)
    assert collision == (CollisionType.SIDE, 2)

    target_agents = np.array([
        ([1000.0, 1000.0], [5., 2., 2.], 1.0, 1),
    ], dtype=target_agents_dtype)

    collision = detect_collision(pred_centroid, pred_yaw, pred_extent, target_agents)
    assert collision is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_distance_to_reference_trajectory(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    # [batch_size, 2]
    pred_centroid = torch.tensor([[1, 0], [1, 1], [1.5, 2]], dtype=torch.float32, device=device)
    # [batch_size, num_timestamps, 2]
    ref_traj = torch.tensor([[[0, 0], [1, 0], [2, 0], [3, 0]],
                             [[0, 0], [1, 0], [2, 0], [3, 0]],
                             [[0, 3], [1, 3], [2, 3], [3, 3]]], dtype=torch.float32, device=device)
    # [batch_size,]
    distance = distance_to_reference_trajectory(pred_centroid, ref_traj)
    expected_distance = torch.tensor([0, 1, 1.11803], dtype=torch.float32, device=device)
    assert torch.allclose(distance, expected_distance, atol=1e-4)
