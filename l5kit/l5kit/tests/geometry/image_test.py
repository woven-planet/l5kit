import numpy as np

from l5kit.geometry import crop_rectangle_from_image


def test_crop_rectangle_from_image() -> None:
    # . X X X X 0 X X X
    # . X X X X X X X X
    # . X X X X X X X X
    # . 3 . . . . . . 1
    # . . . . . . . . .
    # . . . . 2 . . . .
    # 0,1,2,3 are the corners, X are 1-value pixels
    # Actual image is 10x larger in both dimensions
    im = np.zeros((60, 90), dtype=np.uint8)
    im[:30, 10:] = 1

    corners = np.array([[0, 60], [30, 80], [50, 40], [30, 10]])
    crop_image = crop_rectangle_from_image(im, corners)

    # Only one corner is in the "1" area
    corner_sum = crop_image[0, 0] + crop_image[0, -1] + crop_image[-1, 0] + crop_image[-1, -1]
    assert corner_sum == 1
    assert 0.5 > crop_image.mean() > 0.4
