import csv
from itertools import chain

import numpy as np


def write_coords_as_csv(
    csv_file_path: str,
    future_coords_offsets: np.ndarray,
    target_availabilities: np.ndarray,
    timestamps: np.ndarray,
    agent_ids: np.ndarray,
) -> None:
    """
    Write coordinates as a csv file

    Args:
        csv_file_path (str): csv path
        future_coords_offsets (np.ndarray): array of size N x future_num_frames x coords of displacements
        target_availabilities (np.ndarray): array of size N x future_num_frames x 1 of (0,1) availability
        timestamps (np.ndarray): array of size N with timestamps int64
        agent_ids (np.ndarray): array for size N with track_ids int64

    Returns:

    """
    assert len(future_coords_offsets) == len(timestamps) == len(agent_ids) == len(target_availabilities)
    assert len(future_coords_offsets.shape) == len(target_availabilities.shape) == 3
    assert future_coords_offsets.shape[:2] == target_availabilities.shape[:2]
    assert len(timestamps.shape) == len(agent_ids.shape) == 1

    future_num_frames = future_coords_offsets.shape[1]

    target_availabilities = target_availabilities[..., 0]  # remove last axis (both XY invalid or valid)

    with open(csv_file_path, "w") as f:
        # the first line of a csv usually contains a key describing the data.
        writer = csv.writer(f, delimiter=",")

        keys = [
            "timestamp",
            "track_id",
            "future_num_frames",
            *chain.from_iterable([["x" + str(i), "y" + str(i)] for i in range(future_num_frames)]),
            *[f"avail_{i}" for i in range(future_num_frames)],
        ]
        writer.writerow(keys)

        for future_coords_offset, target_availability, timestamp, agent_id in zip(
            future_coords_offsets, target_availabilities, timestamps, agent_ids
        ):
            writer.writerow(
                [
                    timestamp,
                    agent_id,
                    future_num_frames,
                    *chain.from_iterable(future_coords_offset),
                    *target_availability,
                ]
            )
