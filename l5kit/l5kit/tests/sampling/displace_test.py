import numpy as np
import pytest

from l5kit.data import ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.geometry import compute_agent_pose, rotation33_as_yaw, transform_points
from l5kit.rasterization import RenderContext, StubRasterizer
from l5kit.sampling.agent_sampling import get_relative_poses


@pytest.fixture(scope="function")
def base_displacement(zarr_dataset: ChunkedDataset, cfg: dict) -> np.ndarray:
    future_num_frames = cfg["model_params"]["future_num_frames"]
    ref_frame = zarr_dataset.frames[0]
    world_from_agent = compute_agent_pose(
        ref_frame["ego_translation"][:2], rotation33_as_yaw((ref_frame["ego_rotation"]))
    )

    future_positions, *_ = get_relative_poses(
        num_frames=future_num_frames,
        frames=zarr_dataset.frames[1: 1 + future_num_frames],
        selected_track_id=None,
        agents=[np.empty(0) for _ in range(future_num_frames)],
        agent_from_world=np.linalg.inv(world_from_agent),
        current_agent_yaw=rotation33_as_yaw(ref_frame["ego_rotation"]),
    )
    return future_positions


# all these params should not have any effect on the displacement (as it is in agent coordinates)
@pytest.mark.parametrize("raster_size", [(100, 100), (100, 50), (200, 200), (50, 50)])
@pytest.mark.parametrize("ego_center", [(0.25, 0.25), (0.75, 0.75), (0.5, 0.5)])
@pytest.mark.parametrize("pixel_size", [(0.25, 0.25), (0.5, 0.5)])
def test_same_displacement(
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        base_displacement: np.ndarray,
        raster_size: tuple,
        ego_center: tuple,
        pixel_size: tuple,
) -> None:
    cfg["raster_params"]["raster_size"] = raster_size
    cfg["raster_params"]["pixel_size"] = np.asarray(pixel_size)
    cfg["raster_params"]["ego_center"] = np.asarray(ego_center)

    render_context = RenderContext(
        np.asarray(raster_size),
        np.asarray(pixel_size),
        np.asarray(ego_center),
        set_origin_to_bottom=cfg["raster_params"]["set_origin_to_bottom"],
    )
    dataset = EgoDataset(cfg, zarr_dataset, StubRasterizer(render_context), )
    data = dataset[0]
    assert np.allclose(data["target_positions"], base_displacement)


def test_coordinates_straight_road(zarr_dataset: ChunkedDataset, cfg: dict) -> None:
    # on a straight road `target_positions` should increase on x only
    render_context = RenderContext(
        np.asarray(cfg["raster_params"]["raster_size"]),
        np.asarray(cfg["raster_params"]["pixel_size"]),
        np.asarray(cfg["raster_params"]["ego_center"]),
        set_origin_to_bottom=cfg["raster_params"]["set_origin_to_bottom"],
    )
    dataset = EgoDataset(cfg, zarr_dataset, StubRasterizer(render_context), )

    # get first prediction and first 50 centroids
    centroids = []
    preds = []
    preds_world = []
    for idx in range(50):
        data = dataset[idx]
        if idx == 0:
            preds = data["target_positions"]
            preds_world = transform_points(preds, np.linalg.inv(data["agent_from_world"]))

        centroids.append(data["centroid"][:2])
    centroids = np.stack(centroids)

    # compute XY variances for preds and centroids
    var_preds = np.var(preds, 0, ddof=1)
    var_centroids = np.var(centroids, 0, ddof=1)

    assert var_preds[1] / var_preds[0] < 0.001  # variance on Y is way lower than on X
    assert var_centroids[1] / var_centroids[0] > 0.9  # variance on Y is similar to X

    # check similarity between coordinates
    assert np.allclose(preds_world[:-1], centroids[1:])
