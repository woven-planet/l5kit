import os
from typing import List

import numpy as np

from . import FRAME_DTYPE, SCENE_DTYPE, ChunkedDataset


def zarr_concat(input_zarrs: List[str], output_zarr: str, verbose: bool = False) -> None:
    """
    Concat many zarr into a single one. Takes care of updating indices for frames and agents.
    The output zarr can also already exists. In that case, new data is appended

    Args:
        input_zarrs (List[str]): a list of paths to input zarrs
        output_zarr (str): the path to the output zarr
        verbose (bool): chatty output

    Returns:

    """

    output_dataset = ChunkedDataset(output_zarr)
    if os.path.exists(output_zarr):
        output_dataset.open("a")
    else:
        output_dataset.initialize()

    for input_zarr in input_zarrs:

        input_dataset = ChunkedDataset(input_zarr)
        input_dataset.open()

        if verbose:
            print(f"input scenes size: {input_dataset.scenes.shape[0]}")
            print(f"input frames size: {input_dataset.frames.shape[0]}")
            print(f"input agents size: {input_dataset.agents.shape[0]}")

        frame_offset = output_dataset.frames.shape[0]
        new_scenes = np.zeros(input_dataset.scenes.shape[0], dtype=SCENE_DTYPE)

        for i, scene in enumerate(input_dataset.scenes):  # add new scenes to zarr
            scene["frame_index_interval"] = scene["frame_index_interval"] + frame_offset
            new_scenes[i] = scene
        output_dataset.scenes.append(new_scenes)

        agent_offset = output_dataset.agents.shape[0]
        new_frames = np.zeros(input_dataset.frames.shape[0], dtype=FRAME_DTYPE)
        for i, frame in enumerate(input_dataset.frames):  # add new frames to the zarr
            frame["agent_index_interval"] = frame["agent_index_interval"] + agent_offset
            new_frames[i] = frame
        output_dataset.frames.append(new_frames)

        output_dataset.agents.append(input_dataset.agents)  # add new agents to the zarr

    if verbose:
        print(f"output scenes size: {output_dataset.scenes.shape[0]}")
        print(f"output frames size: {output_dataset.frames.shape[0]}")
        print(f"output agents size: {output_dataset.agents.shape[0]}")
