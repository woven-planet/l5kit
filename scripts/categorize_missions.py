import argparse
import csv
import os
from collections import Counter
from typing import Dict, Optional

import numpy as np
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.geometry import rotation33_as_yaw




def identify_turn(zarr_dataset: ChunkedDataset,
                  max_frame_id: Optional[int] = None,
                  max_num_scenes: Optional[int] = None) -> Dict[int, str]:
    """Map each scene to its type based on turning in given zarr_dataset.

    :param zarr_dataset: the dataset
    :param max_frame_id: the maximum id of frame to categorize.
                         Train data has shorter frame lengths.
    :param max_num_scenes: the maximum number of scenes to categorize
    :return: the dict mapping the scene id to its type.
    """
    num_scenes = max_num_scenes or len(zarr_dataset.scenes)
    scenes = zarr_dataset.scenes[:num_scenes]

    mission_dict: Dict[int, str] = {}
    start_ref = zarr_dataset.scenes[0]["start_time"]
    diff_thresh = 50519692288
    mission_id = 0
    mission_lengths = []
    current_mission_length = 0
    gap_lengths = []

    # Loop Over Scenes
    for scene_id, scene in enumerate(scenes):
        current_mission_length += 1
        diff_start_time = scene["start_time"] - start_ref
        start_ref = scene["start_time"]

        if diff_start_time > 2000 * diff_thresh:
            mission_id += 1
            mission_lengths.append(current_mission_length)
            current_mission_length = 0   
            gap_lengths.append(diff_start_time // 1e13)

        mission_dict[scene_id] = str(mission_id)
    
    print("Number of Missions: ", mission_id)
    print("Mission Lengths: ", mission_lengths)
    print("Missions Gaps: ", gap_lengths)
    return mission_dict


# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
os.environ["L5KIT_DATA_FOLDER"] = os.environ["HOME"] + '/level5_data/'
if "L5KIT_DATA_FOLDER" not in os.environ:
    raise KeyError("L5KIT_DATA_FOLDER environment variable not set")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='scenes/sample.zarr',
                        help='Path to L5Kit dataset to categorize')
    parser.add_argument('--output', type=str, default='sample_metadata.csv',
                        help='CSV file name for writing the metadata')
    args = parser.parse_args()

    # load dataset
    dm = LocalDataManager()
    dataset_path = dm.require(args.data_path)
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()

    # categorize
    mission_dict = identify_turn(zarr_dataset)
    categories_counter = Counter(mission_dict.values())
    print("The number of scenes per category:")
    print(categories_counter)

    # Write to csv
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        for key, value in mission_dict.items():
            writer.writerow([key, value])
