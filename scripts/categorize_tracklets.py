import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.geometry import rotation33_as_yaw
from l5kit.rasterization import build_rasterizer
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def subset_and_subsample(dataset: EgoDataset, ratio: float, step: int) -> Subset:
    frames = dataset.dataset.frames
    frames_to_use = range(0, int(ratio * len(frames)), step)

    scene_samples = [dataset.get_frame_indices(f) for f in frames_to_use]
    scene_samples = np.concatenate(scene_samples).ravel()
    scene_samples = np.sort(scene_samples)
    return Subset(dataset, scene_samples)

# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
if "L5KIT_DATA_FOLDER" not in os.environ:
    raise KeyError("L5KIT_DATA_FOLDER environment variable not set")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='scenes/sample.zarr',
                        help='Path to L5Kit dataset to categorize')
    parser.add_argument('--output', type=str, default='sample_cluster_metadata.csv',
                        help='CSV file name for writing the metadata')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--ratio', type=float, default=1.0)
    args = parser.parse_args()

    # load dataset
    dm = LocalDataManager()
    # get config
    path_l5kit = Path(__file__).parents[1]
    cfg = load_config_data(str(path_l5kit / "examples/dro/drivenet_config.yaml"))
    # rasterisation
    rasterizer = build_rasterizer(cfg, dm)

    # load dataset
    dataset_path = dm.require(args.data_path)
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()
    dataset_original = EgoDataset(cfg, zarr_dataset, rasterizer, perturbation=None)

    # Sub-sample and data loader
    ego_dataset = subset_and_subsample(dataset_original, ratio=args.ratio, step=args.step)
    train_dataloader = DataLoader(ego_dataset, batch_size=1)

    target_array: List[np.ndarray] = []
    for data in tqdm(train_dataloader):
        target_xy = data['target_positions'][0]
        # plt.plot(target_xy[:, 0], target_xy[:, 1], 'y--')
        if len(target_array):
            target_array.append(target_xy)
        else:
            target_array = [target_xy]

    target_array = np.stack(target_array)

    # target_array = target_array.reshape(-1, 24)

    # # Slopes
    # features = []
    # for i in target_array:
    #     diff_x = i[1:,0]-i[:-1,0]
    #     diff_y = i[1:,1]-i[:-1,1]
        
    #     features.append(diff_y/diff_x)
    # features = np.nan_to_num(features, True, 1000)

    # Last point
    features = []
    for i in target_array:
        # r = i[-1,0]**2 + i[-1,1]**2 
        features.append([0.1*(i[-1,0]-i[0,0]) , i[-1,1]-i[0,1] ])
    features = np.stack(features)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    kmeans = KMeans(init="random",n_clusters=10,n_init=10,max_iter=500,random_state=42)
    kmeans.fit(features)
    kmeans_centers = []
    from collections import defaultdict
    means = defaultdict(list)
    for i in range(len(kmeans.labels_)):
        pts = target_array[i]
        means[kmeans.labels_[i]].append(pts)
    for i in means:
        pts = np.mean(np.stack(means[i]),axis=0)
        kmeans_centers.append(pts)
        plt.plot(pts[:,0], pts[:,1])

    plt.savefig('tracklet.png')
    plt.close()

    kmeans_centers = np.stack(kmeans_centers)
    print("In K-Means Fitting")
    kmeans_count = []
    for i in range(len(np.unique(kmeans.labels_))):
        kmeans_count.append(np.count_nonzero(kmeans.labels_ == i))
    print(kmeans_count)

    total_scenes = sum(kmeans_count)
    print(total_scenes)
    group_weight: Dict[int, float] = {}
    group_weight = {k: (total_scenes / v) for k, v in enumerate(kmeans_count) if v > 0}
    print(group_weight)

    print("In Training Label Prediction")
    # load TRAIN dataset
    dataset_path = dm.require("scenes/train.zarr")
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()
    dataset_original = EgoDataset(cfg, zarr_dataset, rasterizer)
    bs = 32
    train_dataloader = DataLoader(dataset_original, batch_size=bs, num_workers=16)

    labels_train: List[int] = [-1] * len(dataset_original)
    i = 0
    for data in tqdm(train_dataloader):
        target_xy = data['target_positions']
        features = (target_xy[:, -1, :] - target_xy[:, 0, :]) * np.array([[0.1, 1]])
        labels_train[i * bs: (i+1) * bs] = kmeans.predict(features)
        i += 1


    labels_train = np.array(labels_train).astype(np.int8)
    kmeans_count = []
    for i in range(len(np.unique(labels_train))):
        kmeans_count.append(np.count_nonzero(labels_train == i))
    print(kmeans_count)

    total_scenes = sum(kmeans_count)
    print(total_scenes)
    group_weight: Dict[int, float] = {}
    group_weight = {k: (total_scenes / v) for k, v in enumerate(kmeans_count) if v > 0}
    print(group_weight)
    sample_weights = [group_weight[label] for label in labels_train if label > -1]
    print(len(sample_weights))

    with open('cluster_means.npy', 'wb') as f:
        np.save(f, kmeans_centers)
        np.save(f, kmeans_count)
        np.save(f, sample_weights)

    ####################################################################################
    # import sklearn
    # from sklearn.cluster import kmeans_plusplus
    # from sklearn.cluster import DBSCAN
    
    # # Calculate seeds from kmeans++
    # # centers_init, indices = kmeans_plusplus(target_array, n_clusters=12, random_state=0)
    # # centers_init = centers_init.reshape(-1, 12, 2)
    # # for center in centers_init:
    # #     plt.plot(center[:, 0], center[:, 1], c='black')

    # # Calculate seeds from DBSCAN
    # clustering = DBSCAN(eps=1, min_samples=3).fit(target_array)

    # labels = clustering.labels_

    # no_clusters = len(np.unique(labels) )
    # no_noise = np.sum(np.array(labels) == -1, axis=0)
    # print(no_clusters, no_noise)
    # cm = plt.get_cmap('gist_rainbow')
    # NUM_COLORS = no_clusters

    # centers_init = clustering.components_
    # centers_init = centers_init.reshape(-1, 12, 2)
    # for ind, center in enumerate(centers_init):
    #     if labels[ind] != -1:
    #         plt.plot(center[:, 0], center[:, 1], c=cm(1.0 * (labels[ind] + 1)/ NUM_COLORS))
    #     else:
    #         plt.plot(center[:, 0], center[:, 1], c='black')

    # plt.savefig('tracklet.png')
    # plt.close()
    # import pdb; pdb.set_trace()
    # # categorize
    # categories_counter = Counter(turn_dict.values())
    # print("The number of scenes per category:")
    # print(categories_counter)

    # Write to csv
    # with open(args.output, 'w') as f:
    #     writer = csv.writer(f)
    #     for key, value in turn_dict.items():
    #         writer.writerow([key, value])