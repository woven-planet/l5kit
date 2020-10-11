import bisect
from copy import deepcopy
from operator import itemgetter
from typing import Any, Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

from .agent import AgentDataset


class MultiAgentDataset(Dataset):
    """Multi-agent dataset focuses on one agent at one frame, providing map + box
    rasterization but only for the current data frame; other agents at the same frame
    are also included up to a limit, provided with their history raw data as well."""

    def __init__(
        self, rast_only_agent_dataset: AgentDataset, history_agent_dataset: AgentDataset, num_neighbors: int = 10,
    ):
        super().__init__()
        self.rast_only_agent_dataset = rast_only_agent_dataset
        self.history_agent_dataset = history_agent_dataset
        self.num_neighbors = num_neighbors

    def __len__(self) -> int:
        return len(self.rast_only_agent_dataset)

    def get_others_dict(self, index: int, ego_dict: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        agent_index = self.rast_only_agent_dataset.agents_indices[index]
        frame_index = bisect.bisect_right(self.rast_only_agent_dataset.cumulative_sizes_agents, agent_index)
        frame_indices = self.rast_only_agent_dataset.get_frame_indices(frame_index)
        assert len(frame_indices) >= 1, frame_indices
        frame_indices = frame_indices[frame_indices != index]

        others_dict = []
        # The centroid of the AV in the current frame in world reference system. Unit is meters
        for idx, agent in zip(  # type: ignore
            frame_indices, Subset(self.history_agent_dataset, frame_indices),
        ):
            agent["dataset_idx"] = idx
            agent["dist_to_ego"] = np.linalg.norm(agent["centroid"] - ego_dict["centroid"], ord=2)
            # TODO in future we can convert history positions via agent + ego transformation matrix
            # TODO and get the normalized version
            del agent["image"]
            others_dict.append(agent)

        others_dict = sorted(others_dict, key=itemgetter("dist_to_ego"))
        others_dict = others_dict[: self.num_neighbors]
        others_len = len(others_dict)

        # have to pad because torch has no ragged tensor
        # https://github.com/pytorch/pytorch/issues/25032
        length_to_pad = self.num_neighbors - others_len
        pad_item = deepcopy(ego_dict)
        pad_item["dataset_idx"] = index
        pad_item["dist_to_ego"] = np.nan  # set to nan so you don't by chance use this
        del pad_item["image"]
        return (others_dict + [pad_item] * length_to_pad, others_len)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rast_dict = self.rast_only_agent_dataset[index]
        ego_dict = self.history_agent_dataset[index]
        others_dict, others_len = self.get_others_dict(index, ego_dict)
        ego_dict["image"] = rast_dict["image"]
        return {
            "ego_dict": ego_dict,
            "others_dict": others_dict,
            "others_len": others_len,
        }
