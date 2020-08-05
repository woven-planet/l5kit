from typing import Callable, Dict, Optional

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from l5kit.data import ChunkedDataset, DataManager, get_combined_scenes
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer


def build_dataloader(
    cfg: Dict,
    split: str,
    data_manager: DataManager,
    dataset_class: Callable,
    rasterizer: Rasterizer,
    perturbation: Optional[Perturbation] = None,
    combine_scenes: bool = False,
) -> DataLoader:
    """
    Function to build a dataloader from a dataset of dataset_class. Note we have to pass rasterizer and
    perturbation as the factory functions for those are likely to change between repos.

    Args:
        cfg (dict): configuration dict
        split (str): this will be used to index the cfg to get the correct datasets (train or val currently)
        data_manager (DataManager): manager for resolving paths
        dataset_class (Callable): a class object (EgoDataset or AgentDataset currently) to build the dataset
        rasterizer (Rasterizer): the rasterizer for the dataset
        perturbation (Optional[Perturbation]): an optional perturbation object
        combine_scenes (bool): if to combine scenes that follow up each other perfectly

    Returns:
        DataLoader: pytorch Dataloader object built with Concat and Sub datasets
    """

    data_loader_cfg = cfg[f"{split}_data_loader"]
    datasets = []
    for dataset_param in data_loader_cfg["datasets"]:
        zarr_dataset_path = data_manager.require(key=dataset_param["key"])
        zarr_dataset = ChunkedDataset(path=zarr_dataset_path)
        zarr_dataset.open()
        if combine_scenes:  # possible future deprecation
            zarr_dataset.scenes = get_combined_scenes(zarr_dataset.scenes)

        #  Let's load the zarr dataset with our dataset.
        dataset = dataset_class(cfg, zarr_dataset, rasterizer, perturbation=perturbation)

        scene_indices = dataset_param["scene_indices"]
        scene_subsets = []

        if dataset_param["scene_indices"][0] == -1:  # TODO replace with empty
            scene_subset = Subset(dataset, np.arange(0, len(dataset)))
            scene_subsets.append(scene_subset)
        else:
            for scene_idx in scene_indices:
                valid_indices = dataset.get_scene_indices(scene_idx)
                scene_subset = Subset(dataset, valid_indices)
                scene_subsets.append(scene_subset)

        datasets.extend(scene_subsets)

    #  Let's concatenate the training scenes into one dataset for the data loader to load from.
    concat_dataset: ConcatDataset = ConcatDataset(datasets)

    #  Initialize the data loader that our training loop will iterate on.
    batch_size = data_loader_cfg["batch_size"]
    shuffle = data_loader_cfg["shuffle"]
    num_workers = data_loader_cfg["num_workers"]
    dataloader = DataLoader(dataset=concat_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
