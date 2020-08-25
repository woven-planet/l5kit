from math import degrees, radians

import pytest

from l5kit.geometry import angular_distance


def test_angular_distance() -> None:
    # test easy cases
    assert 30.0 == pytest.approx(degrees(angular_distance(radians(30.0), 0)), 1e-3)
    assert -30.0 == pytest.approx(degrees(angular_distance(0, radians(30.0))), 1e-3)

    assert 90.0 == pytest.approx(degrees(angular_distance(radians(90), 0)), 1e-3)
    assert -90.0 == pytest.approx(degrees(angular_distance(0, radians(90.0))), 1e-3)

    # test overflowing cases
    assert 0.0 == pytest.approx(degrees(angular_distance(radians(180.0), radians(-180))), 1e-3)
    assert 0.0 == pytest.approx(degrees(angular_distance(radians(180.0), radians(-180))), 1e-3)

    # note this may seem counter-intuitive, but 170 - (-20) is in fact -170
    assert -20 == pytest.approx(degrees(angular_distance(radians(170.0), radians(-170))), 1e-3)
    # in the same way, -170 - 20 yields 170
    assert 20 == pytest.approx(degrees(angular_distance(radians(-170.0), radians(170))), 1e-3)

    assert -120.0 == pytest.approx(degrees(angular_distance(radians(150.0), radians(-90))), 1e-3)
    assert 120.0 == pytest.approx(degrees(angular_distance(radians(-90.0), radians(150))), 1e-3)

    # test > np.pi cases
    assert -120.0 == pytest.approx(degrees(angular_distance(radians(150.0 + 360), radians(-90 - 360))), 1e-3)
    assert 120.0 == pytest.approx(degrees(angular_distance(radians(-90 - 360), radians(150 + 360))), 1e-3)

    assert -20 == pytest.approx(degrees(angular_distance(radians(170.0), radians(-170 - 3 * 360))), 1e-3)
    assert -20 == pytest.approx(degrees(angular_distance(radians(170.0), radians(-170 + 3 * 360))), 1e-3)
    assert -20 == pytest.approx(degrees(angular_distance(radians(170.0 + 5 * 360), radians(-170))), 1e-3)
    assert -20 == pytest.approx(degrees(angular_distance(radians(170.0 - 5 * 360), radians(-170))), 1e-3)
