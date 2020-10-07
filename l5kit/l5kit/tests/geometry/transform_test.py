import numpy as np
import pytest
import transforms3d

from l5kit.geometry import transform_point, transform_points, transform_points_batch


def test_transform_batch_points() -> None:
    # transform batch and singular elements one by one, results should match
    # note: we use random here as the validity of transform is checked below

    tfs = np.random.randn(16, 3, 3)
    batch_points = np.random.randn(16, 50, 2)
    output_points = transform_points_batch(batch_points, tfs)

    expected_points = []
    for points, tf in zip(batch_points, tfs):
        expected_points.append(transform_points(points, tf))
    expected_points = np.stack(expected_points)
    assert np.allclose(output_points, expected_points, atol=1e-5)


def test_transform_points() -> None:
    tf = np.asarray([[1.0, 0, 100], [0, 0.5, 50], [0, 0, 1]])

    points = np.array([[0, 10], [10, 0], [10, 10]])
    expected_point = np.array([[100, 55], [110, 50], [110, 55]])

    output_points = transform_points(points, tf)

    np.testing.assert_array_equal(output_points, expected_point)


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
