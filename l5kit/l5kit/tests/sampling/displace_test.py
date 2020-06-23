import numpy as np
import pytest

from l5kit.configs import load_config_data
from l5kit.data import ChunkedStateDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import StubRasterizer


@pytest.fixture(scope="module")
def zarr_dataset() -> ChunkedStateDataset:
    zarr_dataset = ChunkedStateDataset(path="./l5kit/tests/data/single_scene.zarr")
    zarr_dataset.open()
    return zarr_dataset


@pytest.fixture(scope="module")
def base_displacement(zarr_dataset: ChunkedStateDataset) -> np.ndarray:
    cfg = load_config_data("./l5kit/configs/default.yaml")
    cfg["raster_params"]["raster_size"] = (100, 100)
    cfg["raster_params"]["ego_center"] = np.asarray((0.5, 0.5))
    cfg["raster_params"]["pixel_size"] = np.asarray((0.25, 0.25))

    dataset = EgoDataset(
        cfg,
        zarr_dataset,
        StubRasterizer(
            cfg["raster_params"]["raster_size"],
            cfg["raster_params"]["pixel_size"],
            cfg["raster_params"]["ego_center"],
            0.5,
        ),
    )
    data = dataset[0]
    return data["target_positions"]


# all these params should not have any effect on the displacement (as they are in world coordinates)
@pytest.mark.parametrize("raster_size", [(100, 100), (100, 50), (200, 200), (50, 50)])
@pytest.mark.parametrize("ego_center", [(0.25, 0.25), (0.75, 0.75), (0.5, 0.5)])
@pytest.mark.parametrize("pixel_size", [(0.25, 0.25), (0.25, 0.5)])
def test_same_displacement(
    zarr_dataset: ChunkedStateDataset,
    base_displacement: np.ndarray,
    raster_size: tuple,
    ego_center: tuple,
    pixel_size: tuple,
) -> None:
    cfg = load_config_data("./l5kit/configs/default.yaml")
    cfg["raster_params"]["raster_size"] = raster_size
    cfg["raster_params"]["ego_center"] = np.asarray(ego_center)
    cfg["raster_params"]["pixel_size"] = np.asarray(pixel_size)

    dataset = EgoDataset(
        cfg,
        zarr_dataset,
        StubRasterizer(
            cfg["raster_params"]["raster_size"],
            cfg["raster_params"]["pixel_size"],
            cfg["raster_params"]["ego_center"],
            0.5,
        ),
    )
    data = dataset[0]
    assert np.allclose(data["target_positions"], base_displacement)
