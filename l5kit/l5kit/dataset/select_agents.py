import os
import pprint
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

import argparse
import numpy as np
import zarr
from l5kit.data import ChunkedStateDataset, get_combined_scenes
from l5kit.data.filter import _get_label_filter  # TODO expose this without digging
from tqdm import tqdm

from l5kit.data import LocalDataManager

os.environ["BLOSC_NOLOCK"] = "1"  # this is required for multiprocessing


def in_consecutive_frame(frame_idx: int, past_frame_idx: int) -> bool:
    return bool(frame_idx == past_frame_idx + 1)


def in_av_distance(av_translation: np.ndarray, agent_centroid: np.ndarray, th: float) -> bool:
    return bool(np.linalg.norm(av_translation[:2] - agent_centroid) < th)


def in_angular_distance(yaw1: np.ndarray, yaw2: np.ndarray, th: float) -> bool:
    """
    Check if the absolute distance in degrees is under the given threshold
    """
    yaw1_in_deg = np.degrees(yaw1)
    yaw2_in_deg = np.degrees(yaw2)
    assert -180 <= yaw1_in_deg <= 180 and -180 <= yaw2_in_deg <= 180  # ensures the next line gives correct results
    abs_angular_distance = abs((yaw2_in_deg - yaw1_in_deg + 180) % 360 - 180)
    return bool(abs_angular_distance < th)


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


def has_moved(agent1: np.ndarray, agent2: np.ndarray, th: float) -> bool:
    return bool(np.linalg.norm(agent1["centroid"] - agent2["centroid"]) > th)


def get_missing_frame_num(els_drop: List, agents_selected_mask: np.ndarray) -> int:
    """
    check if what has been dropped has been already taken or not
    """
    num = 0
    for _, global_agent_idx, _ in els_drop:
        if not agents_selected_mask[global_agent_idx]:
            num += 1
    return num


def get_valid_agents(
    frames_range: np.ndarray,
    dataset: ChunkedStateDataset,
    th_frames_past: int,
    th_frames_future: int,
    th_agent_filter_probability_threshold: float,
    th_yaw_degree: float,
    th_extent_ratio: float,
    th_movement: float,
    th_distance_av: float,
) -> Tuple[np.ndarray, Counter, tuple]:
    """
    Three types of filters are implemented:
    POINT-WISE: only the current state is considered
    COUPLE-WISE: 2 states considered (new and last added)
    SEQUENCE-WISE: potentially all states are considered

    Return a boolean np.array with the same shape of agents and a counter of report
    """
    frames = dataset.frames[slice(*frames_range)]
    agents_range_start = frames[0]["agent_index_interval"][0]
    agents_range_end = frames[-1]["agent_index_interval"][1]

    agents = dataset.agents[agents_range_start:agents_range_end]

    agents_dict = defaultdict(list)
    agents_selected_mask = np.zeros(len(agents), dtype=np.bool)
    report: Counter = Counter()

    # filter here for point-wise to speed up
    of_interest = _get_label_filter(agents["label_probabilities"], th_agent_filter_probability_threshold)
    global_agent_idx = -1
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx]
        agents_frame = agents[slice(*(frame["agent_index_interval"] - agents_range_start))]

        for agent in agents_frame:
            global_agent_idx += 1
            # store info we need for couple and sequence filters
            agents_dict[agent["track_id"]].append((frame_idx, global_agent_idx, agent))

            # ==== POINT-WISE FILTERS
            if not of_interest[global_agent_idx]:
                frame_lost = get_missing_frame_num(agents_dict[agent["track_id"]], agents_selected_mask)
                report["reject_th_agent_filter_probability_threshold"] += frame_lost
                agents_dict[agent["track_id"]] = []
                continue
            if not in_av_distance(frame["ego_translation"], agent["centroid"], th_distance_av):
                frame_lost = get_missing_frame_num(agents_dict[agent["track_id"]], agents_selected_mask)
                report["reject_th_agent_distance_av_threshold"] += frame_lost
                agents_dict[agent["track_id"]] = []
                continue

            # ==== COUPLE-WISE FILTERS
            if len(agents_dict[agent["track_id"]]) > 1:
                p_frame_idx, p_global_agent_idx, p_agent = agents_dict[agent["track_id"]][-2]  # get prev element
                frame_lost = get_missing_frame_num(agents_dict[agent["track_id"]][:-1], agents_selected_mask)

                if not in_consecutive_frame(frame_idx, p_frame_idx):
                    report["reject_th_hole"] += frame_lost
                    agents_dict[agent["track_id"]] = agents_dict[agent["track_id"]][-1:]
                    continue
                if not in_angular_distance(p_agent["yaw"], agent["yaw"], th_yaw_degree):
                    report["reject_th_yaw_degree"] += frame_lost
                    agents_dict[agent["track_id"]] = agents_dict[agent["track_id"]][-1:]
                    continue
                if not in_extent_ratio(p_agent["extent"], agent["extent"], th_extent_ratio):
                    report["reject_th_extent_ratio"] += frame_lost
                    agents_dict[agent["track_id"]] = agents_dict[agent["track_id"]][-1:]
                    continue

            if len(agents_dict[agent["track_id"]]) == th_frames_future + th_frames_past + 1:
                ref_frame_idx, ref_global_agent_idx, ref_agent = agents_dict[agent["track_id"]][th_frames_past]

                # ==== SEQUENCE-WISE FILTERS
                if not has_moved(agent, agents_dict[agent["track_id"]][0][-1], th_movement):
                    frame_lost = get_missing_frame_num(agents_dict[agent["track_id"]][:1], agents_selected_mask)
                    report["reject_th_movement"] += frame_lost
                    agents_dict[agent["track_id"]] = agents_dict[agent["track_id"]][1:]
                    continue

                # all test passed, agent is add to the final output and sequence advanced
                agents_selected_mask[ref_global_agent_idx] = True

                frame_lost = get_missing_frame_num(agents_dict[agent["track_id"]][:1], agents_selected_mask)
                report["reject_th_num_frames"] += frame_lost
                agents_dict[agent["track_id"]] = agents_dict[agent["track_id"]][1:]

    # compute rejected because of insufficient frames
    for el in agents_dict.values():
        report["reject_th_num_frames"] += get_missing_frame_num(el, agents_selected_mask)

    report["total_reject"] = sum([v for v in report.values()])
    report["total_agent_frames"] = len(agents_selected_mask)
    report["selected_agent_frames"] = int(agents_selected_mask.sum())
    return agents_selected_mask, report, (agents_range_start, agents_range_end)


def select_agents(
    input_folder: str,
    th_agent_prob: float,
    th_history_num_frames: int,
    th_future_num_frames: int,
    th_yaw_degree: float,
    th_extent_ratio: float,
    th_movement: float,
    th_distance_av: float,
    num_workers: int,
) -> None:
    """
    Filter agents from zarr INPUT_FOLDER according to multiple thresholds and store a boolean array of the same shape.
    """
    assert th_future_num_frames > 0

    # ===== LOAD
    dm = LocalDataManager()
    input_folder = dm.require(input_folder)

    zarr_dataset = ChunkedStateDataset(path=input_folder)
    zarr_dataset.open()
    zarr_dataset.scenes = get_combined_scenes(zarr_dataset.scenes)

    output_group = f"{th_history_num_frames}_{th_future_num_frames}_{th_agent_prob}"
    if "agents_mask" in zarr_dataset.root and f"agents_mask/{output_group}" in zarr_dataset.root:
        raise FileExistsError(f"{output_group} exists already! only one is supported for now!")

    frame_index_intervals = zarr_dataset.scenes["frame_index_interval"]

    # build a partial with all args except the first one (will be passed by threads)
    get_valid_agents_partial = partial(
        get_valid_agents,
        dataset=zarr_dataset,
        th_frames_past=th_history_num_frames,
        th_frames_future=th_future_num_frames,
        th_agent_filter_probability_threshold=th_agent_prob,
        th_yaw_degree=th_yaw_degree,
        th_extent_ratio=th_extent_ratio,
        th_movement=th_movement,
        th_distance_av=th_distance_av,
    )

    try:
        root = zarr.open(zarr_dataset.path, mode="a")
        root.create_group("agents_mask")
    except ValueError:
        pass  # group is already there

    agents_mask = zarr.open_array(
        str(Path(zarr_dataset.path) / "agents_mask" / output_group),
        mode="w",
        shape=(len(zarr_dataset.agents),),
        chunks=(10000,),
        dtype=np.bool,
        synchronizer=zarr.ProcessSynchronizer(f"/tmp/ag_mask_{str(uuid4())}.sync"),
    )

    report: Counter = Counter()
    print("starting pool...")
    with Pool(num_workers) as pool:
        tasks = tqdm(enumerate(pool.imap_unordered(get_valid_agents_partial, frame_index_intervals)))
        for idx, (mask, count, agents_range) in tasks:
            report += count
            agents_mask[agents_range[0] : agents_range[1]] = mask
        print("collecting results..")

    assert (
        report["total_agent_frames"] == report["selected_agent_frames"] + report["total_reject"]
    ), "something went REALLY wrong"

    agents_cfg = {
        "th_history_num_frames": th_history_num_frames,
        "th_future_num_frames": th_future_num_frames,
        "th_agent_filter_probability_threshold": th_agent_prob,
        "th_yaw_degree": th_yaw_degree,
        "th_extent_ratio": th_extent_ratio,
        "th_movement": th_movement,
        "th_distance_av": th_distance_av,
    }
    # print report
    pp = pprint.PrettyPrinter(indent=4)
    print(f"start report for {input_folder}")
    pp.pprint({**agents_cfg, **report})
    print(f"end report for {input_folder}")
    print("==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", nargs="+", type=str, required=True, help="zarr path")
    parser.add_argument("--th_agent_prob", type=float, default=0.5, help="perception threshold on agents of interest")
    parser.add_argument("--th_history_num_frames", type=int, default=0, help="frames in the past to be valid")
    parser.add_argument("--th_future_num_frames", type=int, default=12, help="frames in the future to be valid")
    parser.add_argument("--th_yaw_degree", type=float, default=30, help="max absolute distance in degree")
    parser.add_argument("--th_extent_ratio", type=float, default=1.1, help="max change in area allowed")
    parser.add_argument("--th_movement", type=float, default=3, help="threshold on the movement in meters")
    parser.add_argument("--th_distance_av", type=float, default=50, help="threshold on distance from AV in meters")
    parser.add_argument("-j", type=int, default=8, help="number of workers")
    args = parser.parse_args()

    for input_folder in args.input_folder:
        select_agents(
            input_folder,
            args.th_agent_prob,
            args.th_history_num_frames,
            args.th_future_num_frames,
            args.th_yaw_degree,
            args.th_extent_ratio,
            args.th_movement,
            args.th_distance_av,
            args.j,
        )
