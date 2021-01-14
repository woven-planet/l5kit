from typing import Optional

import numpy as np
from tqdm import tqdm

from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.geometry import transform_points
from l5kit.rasterization import RenderContext, StubRasterizer

from .csv_utils import write_gt_csv


def export_zarr_to_csv(
        zarr_dataset: ChunkedDataset,
        csv_file_path: str,
        future_num_frames: int,
        filter_agents_threshold: float,
        step_time: float = 0.1,
        agents_mask: Optional[np.array] = None,
) -> None:
    """Produces a csv file containing the ground truth from a zarr file.

    Arguments:
        zarr_dataset (np.ndarray): The open zarr dataset.
        csv_file_path (str): File path to write a CSV to.
        future_num_frames (int): Amount of future displacements we want.
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        agents_mask (Optional[np.array]): a boolean mask of shape (len(zarr_dataset.agents)) which will be used
        instead of computing the agents_mask
    """

    cfg = {
        "raster_params": {
            "pixel_size": np.asarray((0.25, 0.25)),
            "raster_size": (100, 100),
            "filter_agents_threshold": filter_agents_threshold,
            "disable_traffic_light_faces": True,
            "ego_center": np.asarray((0.5, 0.5)),
            "set_origin_to_bottom": True,
        },
        "model_params": {"history_num_frames": 0, "future_num_frames": future_num_frames, "step_time": step_time},
    }

    render_context = RenderContext(
        np.asarray(cfg["raster_params"]["raster_size"]),
        cfg["raster_params"]["pixel_size"],
        cfg["raster_params"]["ego_center"],
        cfg["raster_params"]["set_origin_to_bottom"],
    )
    rasterizer = StubRasterizer(render_context)
    dataset = AgentDataset(cfg=cfg, zarr_dataset=zarr_dataset, rasterizer=rasterizer, agents_mask=agents_mask)

    future_coords_offsets = []
    target_availabilities = []

    timestamps = []
    agent_ids = []

    for el in tqdm(dataset, desc="extracting GT"):  # type: ignore
        # convert agent coordinates to world offsets
        offsets = transform_points(el["target_positions"], el["world_from_agent"]) - el["centroid"][:2]
        future_coords_offsets.append(offsets)

        timestamps.append(el["timestamp"])
        agent_ids.append(el["track_id"])
        target_availabilities.append(el["target_availabilities"])

    write_gt_csv(
        csv_file_path,
        np.asarray(timestamps),
        np.asarray(agent_ids),
        np.asarray(future_coords_offsets),
        np.asarray(target_availabilities),
    )
