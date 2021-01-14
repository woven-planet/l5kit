import argparse
import multiprocessing
import os
import pprint
import sys
import warnings
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import cpu_count, Pool
from pathlib import Path
from tempfile import gettempdir
from typing import Tuple
from uuid import uuid4

import numpy as np
import zarr
from prettytable import PrettyTable
from tqdm import tqdm

from l5kit.data import ChunkedDataset
from l5kit.data.filter import _get_label_filter  # TODO expose this without digging
from l5kit.geometry import angular_distance


if sys.platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)  # this fixes loop in python 3.8 on MacOS

if sys.platform not in ["win32", "cygwin"]:
    os.environ["BLOSC_NOLOCK"] = "1"  # this is required for multiprocessing
else:
    warnings.warn(
        "Windows detected. BLOSC_NOLOCK has not been set as it causes memory leaks on Windows."
        "However, writing the mask with this config may be inconsistent."
    )

TH_YAW_DEGREE = 30
TH_EXTENT_RATIO = 1.1
TH_DISTANCE_AV = 50


def in_consecutive_frame(frame_idx: int, past_frame_idx: int) -> bool:
    return bool(frame_idx == past_frame_idx + 1)


def in_av_distance(av_translation: np.ndarray, agent_centroid: np.ndarray, th: float) -> bool:
    return bool(np.linalg.norm(av_translation[:2] - agent_centroid) < th)


def in_angular_distance(yaw1: np.ndarray, yaw2: np.ndarray, th: float) -> bool:
    """
    Check if the absolute distance in degrees is under the given threshold
    """

    abs_angular_distance_degrees = abs(angular_distance(yaw2, yaw1)) * 180 / np.pi
    return bool(abs_angular_distance_degrees < th)


def in_extent_ratio(extent1: np.ndarray, extent2: np.ndarray, th: float) -> bool:
    """
    Compute the two areas and then the ratio. The ratio is in the range [1, inf)
    """
    area_1 = extent1[0] * extent1[1]
    area_2 = extent2[0] * extent2[1]
    if area_1 < 0.01 or area_2 < 0.01:  # these are clearly errors (less than 1cm2)
        return False
    ratio = area_1 / area_2 if area_1 > area_2 else area_2 / area_1
    return bool(ratio < th)


def update_mask(mask: np.ndarray, agent_list: list) -> None:
    for idx_el, (frame_idx, mask_idx, agent) in enumerate(agent_list):
        mask[mask_idx][0] = idx_el  # past information
        mask[mask_idx][1] = len(agent_list) - idx_el - 1  # future information


def get_valid_agents(
        frames_range: np.ndarray,
        dataset: ChunkedDataset,
        th_agent_filter_probability_threshold: float,
        th_yaw_degree: float,
        th_extent_ratio: float,
        th_distance_av: float,
) -> Tuple[np.ndarray, Counter, tuple]:
    """
    Two types of filters are implemented:
    POINT-WISE: only the current state is considered
    COUPLE-WISE: 2 states considered (new and last added)

    Return a boolean np.array with the same shape of agents and a counter of report
    """
    frames = dataset.frames[slice(*frames_range)]
    agents_range_start = frames[0]["agent_index_interval"][0]
    agents_range_end = frames[-1]["agent_index_interval"][1]

    agents = dataset.agents[agents_range_start:agents_range_end]
    frames["agent_index_interval"] -= agents_range_start  # sync frame and agents again

    agents_dict = defaultdict(list)

    # for every agent in the .zarr -> (available_past_frame, available_future_frames) using this agent_threshold
    # this means a single mask can be used to generate all configurations of future and past frames
    agents_mask = np.zeros((len(agents), 2), dtype=np.uint32)

    report: Counter = Counter()

    # filter here for point-wise to speed up
    of_interest = _get_label_filter(agents["label_probabilities"], th_agent_filter_probability_threshold)
    global_agent_idx = -1
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx]
        agents_frame = agents[slice(*(frame["agent_index_interval"]))]

        for agent in agents_frame:
            global_agent_idx += 1
            # store info we need for couple and sequence filters
            agents_dict[agent["track_id"]].append((frame_idx, global_agent_idx, agent))

            # ==== POINT-WISE FILTERS
            if not of_interest[global_agent_idx]:
                update_mask(agents_mask, agents_dict[agent["track_id"]][:-1])
                report["reject_th_agent_filter_probability_threshold"] += 1
                agents_dict[agent["track_id"]] = []
                continue

            if not in_av_distance(frame["ego_translation"], agent["centroid"], th_distance_av):
                update_mask(agents_mask, agents_dict[agent["track_id"]][:-1])
                report["reject_th_AV_distance"] += 1
                agents_dict[agent["track_id"]] = []
                continue

            # ==== COUPLE-WISE FILTERS
            if len(agents_dict[agent["track_id"]]) > 1:
                p_frame_idx, p_global_agent_idx, p_agent = agents_dict[agent["track_id"]][-2]  # get prev element

                if not in_consecutive_frame(frame_idx, p_frame_idx):
                    update_mask(agents_mask, agents_dict[agent["track_id"]][:-1])
                    report["reject_th_hole"] += 1
                    agents_dict[agent["track_id"]] = agents_dict[agent["track_id"]][-1:]
                    continue
                if not in_angular_distance(p_agent["yaw"], agent["yaw"], th_yaw_degree):
                    update_mask(agents_mask, agents_dict[agent["track_id"]][:-1])
                    report["reject_th_yaw"] += 1
                    agents_dict[agent["track_id"]] = agents_dict[agent["track_id"]][-1:]
                    continue
                if not in_extent_ratio(p_agent["extent"], agent["extent"], th_extent_ratio):
                    update_mask(agents_mask, agents_dict[agent["track_id"]][:-1])
                    report["reject_th_extent"] += 1
                    agents_dict[agent["track_id"]] = agents_dict[agent["track_id"]][-1:]
                    continue

    # update what is left inside the dict
    for track_id, agent_list in agents_dict.items():
        update_mask(agents_mask, agent_list)
        agents_dict[track_id] = []

    report["total_reject"] = sum([v for v in report.values()])
    report["total_agent_frames"] = len(agents_mask)
    return agents_mask, report, (agents_range_start, agents_range_end)


def select_agents(
        zarr_dataset: ChunkedDataset,
        th_agent_prob: float,
        th_yaw_degree: float,
        th_extent_ratio: float,
        th_distance_av: float,
) -> None:
    """
    Filter agents from zarr INPUT_FOLDER according to multiple thresholds and store a boolean array of the same shape.
    """
    agents_mask_path = Path(zarr_dataset.path) / f"agents_mask/{th_agent_prob}"

    if agents_mask_path.exists():
        raise FileExistsError(f"{th_agent_prob} exists already! only one is supported!")

    frame_index_intervals = zarr_dataset.scenes["frame_index_interval"]

    # build a partial with all args except the first one (will be passed by threads)
    get_valid_agents_partial = partial(
        get_valid_agents,
        dataset=zarr_dataset,
        th_agent_filter_probability_threshold=th_agent_prob,
        th_yaw_degree=th_yaw_degree,
        th_extent_ratio=th_extent_ratio,
        th_distance_av=th_distance_av,
    )

    try:
        root = zarr.open(zarr_dataset.path, mode="a")
        root.create_group("agents_mask")
    except ValueError:
        pass  # group is already there

    agents_mask = zarr.open_array(
        str(agents_mask_path),
        mode="w",
        shape=(len(zarr_dataset.agents), 2),
        chunks=(10000,),
        dtype=np.uint32,
        synchronizer=zarr.ProcessSynchronizer(Path(gettempdir()) / f"ag_mask_{str(uuid4())}.sync"),
    )

    report: Counter = Counter()
    print("starting pool...")
    with Pool(cpu_count()) as pool:
        tasks = tqdm(enumerate(pool.imap_unordered(get_valid_agents_partial, frame_index_intervals)))
        for idx, (mask, count, agents_range) in tasks:
            report += count
            agents_mask[agents_range[0]: agents_range[1]] = mask
            tasks.set_description(f"{idx + 1}/{len(frame_index_intervals)}")
        print("collecting results..")

    agents_cfg = {
        "th_agent_filter_probability_threshold": th_agent_prob,
        "th_yaw_degree": th_yaw_degree,
        "th_extent_ratio": th_extent_ratio,
        "th_distance_av": th_distance_av,
    }
    # print report
    pp = pprint.PrettyPrinter(indent=4)
    print(f"start report for {zarr_dataset.path}")
    pp.pprint({**agents_cfg, **report})

    future_steps = [0, 10, 30, 50]
    past_steps = [0, 10, 30, 50]
    agents_mask_np = np.asarray(agents_mask)

    table = PrettyTable(field_names=["past/future"] + [str(step) for step in future_steps])
    for step_p in tqdm(past_steps, desc="computing past/future table"):
        row = [step_p]
        for step_f in future_steps:
            past_mask = agents_mask_np[:, 0] >= step_p
            future_mask = agents_mask_np[:, 1] >= step_f
            row.append(np.sum(past_mask * future_mask))
        table.add_row(row)
    print(table)
    print(f"end report for {zarr_dataset.path}")
    print("==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folders", nargs="+", type=str, required=True, help="zarr path")
    parser.add_argument("--th_agent_prob", type=float, required=True, help="perception threshold on agents")
    parser.add_argument("--th_yaw_degree", type=float, default=TH_YAW_DEGREE, help="max absolute distance in degree")
    parser.add_argument("--th_extent_ratio", type=float, default=TH_EXTENT_RATIO, help="max change in area allowed")
    parser.add_argument("--th_distance_av", type=float, default=TH_DISTANCE_AV, help="max distance from AV in meters")
    args = parser.parse_args()

    for input_folder in args.input_folders:
        zarr_dataset = ChunkedDataset(path=input_folder)
        zarr_dataset.open()

        select_agents(
            zarr_dataset, args.th_agent_prob, args.th_yaw_degree, args.th_extent_ratio, args.th_distance_av,
        )
