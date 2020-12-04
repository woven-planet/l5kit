import numpy as np
import pytest
import transforms3d

from l5kit.geometry import transform_point, transform_points


def test_transform_batch_points() -> None:
    # transform batch and singular elements one by one, results should match
    # note: we use random here as the validity of transform is checked below

    tfs = np.random.randn(16, 3, 3)
    batch_points = np.random.randn(16, 50, 2)
    output_points = transform_points(batch_points, tfs)

    expected_points = []
    for points, tf in zip(batch_points, tfs):
        expected_points.append(transform_points(points, tf))
    expected_points = np.stack(expected_points)
    assert np.allclose(output_points, expected_points, atol=1e-5)


def test_transform_points_broadcast() -> None:
    # transform a batch of points with the same matrix, should be the same a transforming each
    tf = np.random.randn(3, 3)
    batch_points = np.random.randn(16, 50, 2)
    output_points = transform_points(batch_points, tf)

    expected_points = []
    for points in batch_points:
        expected_points.append(transform_points(points, tf))
    expected_points = np.stack(expected_points)
    assert np.allclose(output_points, expected_points, atol=1e-5)


def test_transform_points() -> None:
    # 2D points (e.g. "world_from_agent")
    tf = np.asarray(
        [
            [-1.08912448e00, 2.25029062e00, 6.03325482e03],
            [-2.25029062e00, -1.08912448e00, -1.28582624e03],
            [0.0, 0.0, 1.0],
        ]
    )
    points = np.array([[0, 10], [10, 0], [10, 10]])
    expected_points = np.array([[6055.757726, -1296.717485], [6022.363575, -1308.329146], [6044.866481, -1319.220391]])
    output_points = transform_points(points, tf)
    np.testing.assert_allclose(output_points, expected_points)

    # 3D points (e.g. "world_to_ecef")
    tf = np.asarray(
        [
            [0.846617444, 0.323463078, -0.422623402, -2698767.44],
            [-0.532201938, 0.514559352, -0.672301845, -4293151.58],
            [-3.05311332e-16, 0.794103464, 0.6077826, 3855164.76],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    points = np.array([[0, 10, 0], [10, 0, 0], [0, 0, 10], [10, 10, 0], [10, 10, 10], [0, 10, 10]])
    expected_points = np.array(
        [
            [-2698764.20536922, -4293146.43440648, 3855172.70103464],
            [-2698758.97382556, -4293156.90201938, 3855164.76],
            [-2698771.66623402, -4293158.30301845, 3855170.837826],
            [-2698755.73919478, -4293151.75642586, 3855172.70103464],
            [-2698759.9654288, -4293158.47944431, 3855178.77886064],
            [-2698768.43160324, -4293153.15742493, 3855178.77886064],
        ]
    )
    output_points = transform_points(points, tf)
    np.testing.assert_allclose(output_points, expected_points)

    # original test
    tf = np.asarray([[1.0, 0, 100], [0, 0.5, 50], [0, 0, 1]])
    points = np.array([[0, 10], [10, 0], [10, 10]])
    expected_points = np.array([[100, 55], [110, 50], [110, 55]])
    output_points = transform_points(points, tf)
    np.testing.assert_array_equal(output_points, expected_points)


def test_transform_single_point() -> None:
    tf = np.asarray([[1.0, 0, 100], [0, 0.5, 50], [0, 0, 1]])

    point = np.array([0, 10])
    expected_point = np.array([100, 55])

    output_point = transform_point(point, tf)

    np.testing.assert_array_equal(output_point, expected_point)


def test_transform_points_revert_equivalence() -> None:
    input_points = np.random.rand(10, 3)

    #  Generate some random transformation matrix
    tf = np.identity(4)
    tf[:3, :3] = transforms3d.euler.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
    tf[3, :3] = np.random.rand(3)

    output_points = transform_points(input_points, tf)

    tf_inv = np.linalg.inv(tf)

    input_points_recovered = transform_points(output_points, tf_inv)

    np.testing.assert_almost_equal(input_points_recovered, input_points, decimal=10)


def test_wrong_input_shape() -> None:
    tf = np.eye(4)

    with pytest.raises(AssertionError):
        points = np.zeros((3, 10))
        transform_points(points, tf)

    with pytest.raises(AssertionError):
        points = np.zeros((10, 4))  # should be 3D for a 4D matrix
        transform_points(points, tf)

    with pytest.raises(AssertionError):
        points = np.zeros((10, 3))  # should be 2D for a 3D matrix
        transform_points(points, tf[:3, :3])
