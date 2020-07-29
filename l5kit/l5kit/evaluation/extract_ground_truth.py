import numpy as np

from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.evaluation.write_csv import write_coords_as_csv
from l5kit.rasterization import StubRasterizer


def export_zarr_to_ground_truth_csv(
    zarr_dataset: ChunkedDataset,
    csv_file_path: str,
    future_num_frames: int,
    filter_agents_threshold: float,
    history_step_size: int = 1,
    future_step_size: int = 1,
) -> None:
    """Produces a csv file containing the ground truth from a zarr file.

    Arguments:
        zarr_dataset (np.ndarray): The open zarr dataset.
        csv_file_path (str): File path to write a CSV to.
        future_num_frames (int): Amount of future displacements we want.
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        history_step_size (int): Steps to take between frames, can be used to subsample history frames.
        future_step_size (int): Steps to take between targets into the future.
    """

    assert future_step_size == history_step_size == 1, "still not handled in select_agents"
    cfg = {
        "raster_params": {
            "pixel_size": np.asarray((0.25, 0.25)),
            "raster_size": (100, 100),
            "filter_agents_threshold": filter_agents_threshold,
            "ego_center": np.asarray((0.5, 0.5)),
        },
        "model_params": {
            "history_num_frames": 0,
            "future_num_frames": future_num_frames,
            "history_step_size": history_step_size,
            "future_step_size": future_step_size,
        },
    }

    rasterizer = StubRasterizer(
        raster_size=cfg["raster_params"]["raster_size"],
        pixel_size=cfg["raster_params"]["pixel_size"],
        ego_center=cfg["raster_params"]["ego_center"],
        filter_agents_threshold=filter_agents_threshold,
    )
    dataset = AgentDataset(cfg=cfg, zarr_dataset=zarr_dataset, rasterizer=rasterizer)

    future_coords_offsets = []
    timestamps = []
    agent_ids = []

    for el in dataset:  # type: ignore
        future_coords_offsets.append(el["target_positions"])
        timestamps.append(el["timestamp"])
        agent_ids.append(el["track_id"])
        # TODO export availability

    write_coords_as_csv(
        csv_file_path,
        future_num_frames,
        np.asarray(future_coords_offsets),
        np.asarray(timestamps),
        np.asarray(agent_ids),
    )
