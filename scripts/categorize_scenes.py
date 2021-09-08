import argparse
import csv
import os
from collections import Counter
from typing import Dict, Optional

import numpy as np
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.geometry import rotation33_as_yaw

#: Threshold in rads to determine turn
TURN_THRESH = 1.0
#: Threshold in timesteps to determine turn
TIME_DURATION_TURN = 50


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

    turn_dict: Dict[int, str] = {}
    # Loop Over Scenes
    for scene_id, scene_data in enumerate(scenes):
        frame_ids = scene_data["frame_index_interval"]
        start_frame, end_scene_frame = frame_ids[0], frame_ids[1]
        num_frames_in_scene = end_scene_frame - start_frame
        num_frames_to_categorize = max_frame_id or num_frames_in_scene
        end_frame = start_frame + num_frames_to_categorize
        if TIME_DURATION_TURN > num_frames_to_categorize:
            raise ValueError("Numbers of frames for categorization should be greater than \
                             or equal to TIME_DURATION_TURN")
        frames = zarr_dataset.frames[start_frame:end_frame]

        yaws = np.zeros(len(frames),)
        # iterate over frames
        for idx, frame in enumerate(frames):
            yaws[idx] = rotation33_as_yaw(frame["ego_rotation"])

        # Determine Turn
        turn_type = "straight"
        yaw_diff = yaws[TIME_DURATION_TURN:] - yaws[:-TIME_DURATION_TURN]
        if np.sum(yaw_diff >= TURN_THRESH):
            turn_type = "left"
        elif np.sum(yaw_diff <= -TURN_THRESH):
            turn_type = "right"

        # Update dict
        turn_dict[scene_id] = turn_type
    return turn_dict


# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
os.environ["L5KIT_DATA_FOLDER"] = "/home/ubuntu/level5_data"
if "L5KIT_DATA_FOLDER" not in os.environ:
    raise KeyError("L5KIT_DATA_FOLDER environment variable not set")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='scenes/sample.zarr',
                        help='Path to L5Kit dataset to categorize')
    parser.add_argument('--output', type=str, default='sample_metadata.csv',
                        help='CSV filename name for writing the metadata')
    args = parser.parse_args()

    # load dataset
    dm = LocalDataManager()
    dataset_path = dm.require(args.data_path)
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()

    # categorize
    turn_dict = identify_turn(zarr_dataset)
    res = Counter(turn_dict.values())
    print(res)

    # Write to csv
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        for key, value in turn_dict.items():
            writer.writerow([key, value])
