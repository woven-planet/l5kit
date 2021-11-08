from typing import Dict, Iterator, List

import numpy as np
import torch

from l5kit.dataset import EgoDataset
from torch.utils.data import Subset, BatchSampler, Sampler


def get_sample_weights(scene_type_to_id: Dict[str, List[int]], cumulative_sizes: np.array,
                       ratio: float, step: int) -> List[float]:
    """This Sampler first uniformly selects a group-type and then randomly samples a
    frame belonging to that group.

    :param scene_type_to_id: a dict mapping scene types to their corresponding ids
    :param cumulative_sizes: List of frame demarcations of the dataset
    """
    # Determine Group Statistics and Weights
    num_groups = len(scene_type_to_id.keys())
    group_count = {k: len(v) for k, v in scene_type_to_id.items()}
    total_scenes = sum(group_count.values())

    group_weight: Dict[str, float] = {}
    group_weight = {k: (total_scenes / v) for k, v in group_count.items() if v > 0}
    # for k, v in group_count.items():
    #     if v <= 100:
    #         group_weight[k] = 0
    #     else:
    #         group_weight[k] = total_scenes / v

    # Loop over scenes
    total_frames = cumulative_sizes[-1]
    cumulative_sizes = np.insert(cumulative_sizes, 0, 0)
    sample_weights = [0] * total_frames
    for index in range(len(cumulative_sizes)-1):
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

    # Filter according to ratio and step
    frames_to_use = range(0, int(ratio * len(sample_weights)), step)
    sample_weights_filtered = [sample_weights[f] for f in frames_to_use]
    return sample_weights_filtered


def append_reward_scaling(data_batch: Dict[str, torch.Tensor], reward_scale: Dict[str, float],
                          scene_id_to_type_list: List[List[str]]) -> Dict[str, torch.Tensor]:
    """Determine reward scaling for each sample based on the group the sample belongs to.

    :param data_batch: the current data batch
    :param reward_scale: the dict that determines the reward scaling per group
    :param scene_id_to_type_list: a list mapping scene id to their corresponding types
    :return: The updated data_batch with "reward_scaling" key
    """
    reward_scaling = [reward_scale[scene_id_to_type_list[scene_id][0]] for scene_id in data_batch['scene_index']]
    data_batch["reward_scaling"] = torch.FloatTensor(reward_scaling)
    return data_batch


def subset_and_subsample(dataset: EgoDataset, ratio: float, step: int) -> Subset:
    frames = dataset.dataset.frames
    frames_to_use = range(0, int(ratio * len(frames)), step)

    scene_samples = [dataset.get_frame_indices(f) for f in frames_to_use]
    scene_samples = np.concatenate(scene_samples).ravel()
    scene_samples = np.sort(scene_samples)
    return Subset(dataset, scene_samples)


def append_group_index(data_batch: Dict[str, torch.Tensor], group_str: List[str],
                       scene_id_to_type_list: List[List[str]]) -> Dict[str, torch.Tensor]:
    """Determine reward scaling for each sample based on the group the sample belongs to.

    :param data_batch: the current data batch
    :param group_str: the list of group names to identify index
    :param scene_id_to_type_list: a list mapping scene id to their corresponding types
    :return: The updated data_batch with "reward_scaling" key
    """
    group_index = [group_str.index(scene_id_to_type_list[scene_id][0]) for scene_id in data_batch['scene_index']]
    data_batch["group_index"] = torch.IntTensor(group_index)
    return data_batch


def append_group_index_cluster(data_batch: Dict[str, torch.Tensor],
                               cluster_means: np.ndarray) -> Dict[str, torch.Tensor]:
    """Determine reward scaling for each sample based on the cluster group the sample belongs to.

    :param data_batch: the current data batch
    :param cluster_means: the cluster means
    :return: The updated data_batch with "reward_scaling" key
    """
    target_positions = data_batch["target_positions"]
    batch_size = len(target_positions)
    num_cluster = len(cluster_means)
    distance_cluster = target_positions.unsqueeze(1) - cluster_means
    distance_cluster = distance_cluster.view(batch_size, num_cluster, -1)
    distance_cluster = torch.norm(distance_cluster, dim=-1)
    group_index = torch.argmin(distance_cluster, dim=1)
    data_batch["group_index"] = torch.LongTensor(group_index)
    return data_batch


def get_sample_weights_clusters(cluster_sample_wt: np.ndarray,
                                ratio: float, step: int) -> List[float]:
    """This Sampler first uniformly selects a group-type and then randomly samples a
    frame belonging to that group.

    :param scene_type_to_id: a dict mapping scene types to their corresponding ids
    :param cumulative_sizes: List of frame demarcations of the dataset
    """

    # Filter according to ratio and step
    frames_to_use = range(0, int(ratio * len(cluster_sample_wt)), step)
    sample_weights_filtered = [cluster_sample_wt[f] for f in frames_to_use]
    return sample_weights_filtered


class GroupBatchSampler(BatchSampler):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool,
                 scene_type_to_id: Dict[str, List[int]], cumulative_sizes: np.array,
                 step: int, group_size: int = 8) -> None:
            super(GroupBatchSampler, self).__init__(sampler, batch_size, drop_last)
            self.scene_type_to_id = scene_type_to_id
            cumulative_sizes = np.insert(cumulative_sizes, 0, 0)
            self.cumulative_sizes = torch.Tensor(cumulative_sizes).long()
            self.step = step
            self.group_size = group_size

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            list_indices = torch.randint(0, len(self.scene_type_to_id[str(idx)]), (self.group_size,))
            for list_index in list_indices:
                scene_index = self.scene_type_to_id[str(idx)][list_index]
                # Start Frame
                start_frame_index = self.cumulative_sizes[scene_index].item()
                start_frame_index = self.step * (start_frame_index // self.step + 1)
                # End Frame
                end_frame_index = self.cumulative_sizes[scene_index + 1].item()
                end_frame_index = self.step * (end_frame_index // self.step)
                # Get Frame Index
                frame_index = torch.randint(start_frame_index, end_frame_index, (1,)).item()
                frame_index = frame_index // self.step
                batch.append(frame_index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return 8 * len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return 8 * (len(self.sampler) + self.batch_size - 1) // self.batch_size
