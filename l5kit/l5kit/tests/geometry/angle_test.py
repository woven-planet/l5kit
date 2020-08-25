from math import degrees, radians

import numpy as np
import pytest

from l5kit.geometry import angular_distance


def test_angular_distance() -> None:
    # input yaw should be in -pi,pi
    with pytest.raises(AssertionError):
        angular_distance(np.pi + 0.1, 0)
    with pytest.raises(AssertionError):
        angular_distance(0, np.pi + 0.1)
    with pytest.raises(AssertionError):
        angular_distance(-(np.pi + 0.1), 0)
    with pytest.raises(AssertionError):
        angular_distance(0, -(np.pi + 0.1))

    # test easy cases
    assert pytest.approx(degrees(angular_distance(radians(30.0), 0)), 30.0, 1e-3)
    assert pytest.approx(degrees(angular_distance(0, radians(30.0))), -30.0, 1e-3)

    assert pytest.approx(degrees(angular_distance(radians(90), 0)), 90.0, 1e-3)
    assert pytest.approx(degrees(angular_distance(0, radians(90.0))), -90.0, 1e-3)

    # test limits
    assert pytest.approx(degrees(angular_distance(radians(180.0), 0)), 180.0, 1e-3)
    assert pytest.approx(degrees(angular_distance(0, radians(180.0))), -180.0, 1e-3)
    assert pytest.approx(degrees(angular_distance(radians(-180.0), 0)), -180.0, 1e-3)
    assert pytest.approx(degrees(angular_distance(0, radians(-180.0))), 180.0, 1e-3)

    # test overflowing cases
    assert pytest.approx(degrees(angular_distance(radians(180.0), radians(-180))), 0.0, 1e-3)
    assert pytest.approx(degrees(angular_distance(radians(180.0), radians(-180))), 0.0, 1e-3)

    assert pytest.approx(degrees(angular_distance(radians(170.0), radians(-170))), 20.0, 1e-3)
    assert pytest.approx(degrees(angular_distance(radians(-170.0), radians(170))), -20.0, 1e-3)

    assert pytest.approx(degrees(angular_distance(radians(150.0), radians(-90))), 120.0, 1e-3)
    assert pytest.approx(degrees(angular_distance(radians(-90.0), radians(150))), -120.0, 1e-3)
