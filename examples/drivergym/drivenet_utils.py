from typing import Dict, List

import numpy as np
import torch

from l5kit.dataset import EgoDataset
from torch.utils.data import Subset

def get_sample_weights(scene_type_to_id: Dict[str, List[int]], 
                       max_scene_id: int, cumulative_sizes: np.array) -> List[float]:
    """This Sampler first uniformly selects a group-type and then randomly samples a
    frame belonging to that group.

    :param scene_type_to_id: a dict mapping scene types to their corresponding ids
    :param max_scene_id: the maximum scene id to sample
    :param cumulative_sizes: List of frame demarcations of the dataset
    """


    # Determine Group Statistics and Weights
    num_groups = len(scene_type_to_id.keys())
    group_count = {k: len(v) for k, v in scene_type_to_id.items()}
    total_scenes = sum(group_count.values())
    group_weight: Dict[str, float] = {}
    group_weight = {k: (total_scenes / v) for k, v in group_count.items() if v > 0}

    # Loop over scenes
    total_frames = cumulative_sizes[-1]
    cumulative_sizes = np.insert(cumulative_sizes, 0, 0)
    sample_weights = [0] * total_frames
    for index in range(max_scene_id):
        # Determine boundaries
        start_frame = cumulative_sizes[index]
        end_frame = cumulative_sizes[index+1]
        len_scene = end_frame - start_frame

        # Get Scene Type
        scene_type: str
        for group, id_list in scene_type_to_id.items():
            if index in id_list:
                scene_type = group
                break

        # Assign frame weights based on Scene type
        sample_weights[start_frame : end_frame] = [group_weight[scene_type]] * len_scene

    return sample_weights


def append_reward_scaling(data_batch: Dict[str, torch.Tensor], reward_scale: Dict[str, float],
                          scene_type_to_id_dict: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
    """Determine reward scaling for each sample based on the group the sample belongs to.

    :param data_batch: the current data batch
    :param reward_scale: the dict that determines the reward scaling per group
    :param scene_type_to_id: a dict mapping scene types to their corresponding ids
    :return: The updated data_batch with "reward_scaling" key
    """
    reward_scaling = torch.ones_like(data_batch['scene_index'])
    for idx, scene_id in enumerate(data_batch['scene_index']):
        if scene_id in scene_type_to_id_dict["left"]:
            reward_scaling[idx] = reward_scale["left"]
        elif scene_id in scene_type_to_id_dict["right"]:
            reward_scaling[idx] = reward_scale["right"]
        else:
            reward_scaling[idx] = reward_scale["straight"]
    data_batch["reward_scaling"] = reward_scaling
    return data_batch


def subset_and_subsample(dataset: EgoDataset, ratio: float, step: int) -> Subset:
    frames = dataset.dataset.frames
    frames_to_use = range(0, int(ratio * len(frames)), step)

    scene_samples = [dataset.get_frame_indices(f) for f in frames_to_use]
    scene_samples = np.concatenate(scene_samples).ravel()
    scene_samples = np.sort(scene_samples)
    return Subset(dataset, scene_samples)
