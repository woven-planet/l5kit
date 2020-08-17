import argparse
from pathlib import Path

import numpy as np
from zarr import convenience

from l5kit.data import ChunkedDataset
from l5kit.data.zarr_utils import zarr_scenes_chunk
from l5kit.dataset.select_agents import TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE, select_agents
from l5kit.evaluation import export_zarr_to_csv

MIN_FUTURE_STEPS = 10
FUTURE_STEPS = 50


def chop_dataset(
    zarr_path: str, th_agent_prob: float, num_frames_to_copy: int, num_frames_gt: int, min_frame_future: int
) -> str:
    """
    Create a chopped version of the zarr that can be used for evaluation. This function is used to generate tests
    set that have no GT in the data for selected agents.

    Store:
     - a dataset where each scene has been chopped at `num_frames_to_copy` frames;
     - a mask for agents for those final frames based on the original mask and a threshold on the future_frames;
     - the GT csv for those agents

    Args:
        zarr_path (str): zarr path as a string
        th_agent_prob (float): threshold over agents prob used in select_agents
        num_frames_to_copy (int):  how many frames to copy from the beginning of each scene
        min_frame_future (int): min number of frames available in the future for an agent to be taken
        num_frames_gt (int): how many future predictions to store in the GT file

    Returns:
        str: the parent folder of the new data
    """
    zarr_path = Path(zarr_path)
    dest_path = zarr_path.parent / f"{zarr_path.stem}_chopped_{num_frames_to_copy}"
    chopped_path = dest_path / zarr_path.name
    gt_path = dest_path / "gt.csv"
    mask_chopped_path = dest_path / "mask"

    # 1: Create original mask
    zarr_dt = ChunkedDataset(str(zarr_path))
    zarr_dt.open()

    agents_mask_path = Path(zarr_path) / f"agents_mask/{th_agent_prob}"
    if not agents_mask_path.exists():  # don't check in root but check for the path
        select_agents(
            zarr_dt,
            th_agent_prob=th_agent_prob,
            th_yaw_degree=TH_YAW_DEGREE,
            th_extent_ratio=TH_EXTENT_RATIO,
            th_distance_av=TH_DISTANCE_AV,
        )
    agents_mask_origin = np.asarray(convenience.load(str(agents_mask_path)))

    # 2: compute chop
    zarr_scenes_chunk(str(zarr_path), str(chopped_path), num_frames_to_copy=num_frames_to_copy)
    zarr_chopped = ChunkedDataset(str(chopped_path))
    zarr_chopped.open()

    # 3: Compute new mask
    agents_mask_chop_bool = np.zeros(len(zarr_chopped.agents), dtype=np.bool)
    agents_mask_orig_bool = np.zeros(len(zarr_dt.agents), dtype=np.bool)

    for idx in range(len(zarr_dt.scenes)):
        scene = zarr_dt.scenes[idx]

        frame_original = zarr_dt.frames[scene["frame_index_interval"][0] + num_frames_to_copy - 1]
        slice_agents_original = slice(*frame_original["agent_index_interval"])
        frame_chopped = zarr_chopped.frames[zarr_chopped.scenes[idx]["frame_index_interval"][-1] - 1]
        slice_agents_chopped = slice(*frame_chopped["agent_index_interval"])

        # 3.1: filter agents_mask_origin for future
        mask = agents_mask_origin[slice_agents_original][:, 1] >= min_frame_future
        agents_mask_orig_bool[slice_agents_original] = mask.copy()
        agents_mask_chop_bool[slice_agents_chopped] = mask.copy()

    # 4: store information
    np.savez(str(mask_chopped_path), agents_mask_chop_bool)
    export_zarr_to_csv(
        zarr_dt, str(gt_path), num_frames_gt, th_agent_prob, agents_mask=agents_mask_orig_bool,
    )
    return str(dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_paths", required=True, nargs="+", type=Path)
    parser.add_argument("--th_agent_prob", type=float, required=True, help="perception threshold on agents")
    parser.add_argument("--num_frames_to_copy", required=True, type=int)
    parser.add_argument("--future_steps", required=True, type=int)
    parser.add_argument("--min_future_steps", default=MIN_FUTURE_STEPS, type=int)
    args = parser.parse_args()
    for zarr_path in args.zarr_paths:
        chop_dataset(zarr_path, args.th_agent_prob, args.num_frames_to_copy, args.future_steps, args.min_future_steps)
