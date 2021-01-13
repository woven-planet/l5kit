import numpy as np

from l5kit.rasterization.semantic_rasterizer import indices_in_bounds


def test_elements_within_bounds() -> None:
    center = np.zeros(2)
    half_side = 0.5  # square centered around origin with side 1
    bounds = np.zeros((1, 2, 2))

    # non-intersecting
    bounds[0, 0] = (2, 2)
    bounds[0, 1] = (4, 4)
    assert len(indices_in_bounds(center, bounds, half_side)) == 0

    # intersecting only x
    bounds[0, 0] = (0, 2)
    bounds[0, 1] = (4, 4)
    assert len(indices_in_bounds(center, bounds, half_side)) == 0

    # intersecting only y
    bounds[0, 0] = (2, 0)
    bounds[0, 1] = (4, 4)
    assert len(indices_in_bounds(center, bounds, half_side)) == 0

    # intersecting both with min (valid)
    bounds[0, 0] = (0.25, 0.25)
    bounds[0, 1] = (4, 4)
    assert len(indices_in_bounds(center, bounds, half_side)) == 1

    # intersecting both with max (valid)
    bounds[0, 0] = (-4, -4)
    bounds[0, 1] = (0.25, 0.25)
    assert len(indices_in_bounds(center, bounds, half_side)) == 1

    # inside (valid)
    bounds[0, 0] = (-0.25, -0.25)
    bounds[0, 1] = (0.25, 0.25)
    assert len(indices_in_bounds(center, bounds, half_side)) == 1

    # including (valid)
    bounds[0, 0] = (-4, -4)
    bounds[0, 1] = (4, 4)
    assert len(indices_in_bounds(center, bounds, half_side)) == 1

    # empty case
    bounds = np.empty((0, 2, 2), dtype=np.float32)
    assert len(indices_in_bounds(center, bounds, half_side)) == 0
