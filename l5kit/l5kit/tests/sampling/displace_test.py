import numpy as np
import pytest

from l5kit.data import ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.geometry import rotation33_as_yaw
from l5kit.rasterization import StubRasterizer
from l5kit.sampling.agent_sampling import _create_targets_for_deep_prediction


@pytest.fixture(scope="function")
def base_displacement(zarr_dataset: ChunkedDataset, cfg: dict) -> np.ndarray:
    future_num_frames = cfg["model_params"]["future_num_frames"]
    ref_frame = zarr_dataset.frames[0]
    future_coords_offset, *_ = _create_targets_for_deep_prediction(
        num_frames=future_num_frames,
        frames=zarr_dataset.frames[1 : 1 + future_num_frames],
        selected_track_id=None,
        agents=[np.empty(0) for _ in range(future_num_frames)],
        agent_current_centroid=ref_frame["ego_translation"][:2],
        agent_current_yaw=rotation33_as_yaw(ref_frame["ego_rotation"]),
    )
    return future_coords_offset


# all these params should not have any effect on the displacement (as it is in world coordinates)
@pytest.mark.parametrize("raster_size", [(100, 100), (100, 50), (200, 200), (50, 50)])
@pytest.mark.parametrize("ego_center", [(0.25, 0.25), (0.75, 0.75), (0.5, 0.5)])
@pytest.mark.parametrize("pixel_size", [(0.25, 0.25), (0.25, 0.5)])
def test_same_displacement(
    cfg: dict,
    zarr_dataset: ChunkedDataset,
    base_displacement: np.ndarray,
    raster_size: tuple,
    ego_center: tuple,
    pixel_size: tuple,
) -> None:
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
