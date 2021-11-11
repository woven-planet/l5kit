import argparse
import csv
import os
from collections import Counter
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional

import numpy as np
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.geometry import rotation33_as_yaw

#: Threshold in rads to determine turn
TURN_THRESH = 1.0
#: Threshold in timesteps to determine turn
TIME_DURATION_TURN = 50


def identify_drivers(zarr_dataset: ChunkedDataset,
                     max_num_scenes: Optional[int] = None) -> Dict[int, str]:
    """Map each scene to its type based on turning in given zarr_dataset.

    :param zarr_dataset: the dataset
    :param max_num_scenes: the maximum number of scenes to categorize
    :return: the dict mapping the scene id to its driver.
    """
    num_scenes = max_num_scenes or len(zarr_dataset.scenes)
    scenes = zarr_dataset.scenes[:num_scenes]

    driver_dict: DefaultDict[str, List[int]] = defaultdict(list)
    # Loop Over Scenes
    for scene_id, scene_data in enumerate(scenes):
        driver_dict[scene_data["host"]].append(scene_id)
    return driver_dict


# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
os.environ["L5KIT_DATA_FOLDER"] = os.environ["HOME"] + '/level5_data/'
if "L5KIT_DATA_FOLDER" not in os.environ:
    raise KeyError("L5KIT_DATA_FOLDER environment variable not set")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='scenes/train_full.zarr',
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
    driver_dict = identify_drivers(zarr_dataset)
    print(len(driver_dict), driver_dict.keys())
    import pdb; pdb.set_trace()
    # categories_counter = Counter(driver_dict.values())
    # print("The number of scenes per category:")
    # print(categories_counter)

    # Write to csv
    # with open(args.output, 'w') as f:
    #     writer = csv.writer(f)
    #     for key, value in turn_dict.items():
    #         writer.writerow([key, value])