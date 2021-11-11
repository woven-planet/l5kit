
import os

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.data import get_dataset_path

def load_data(dataset_name="train_data_loader"):


    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"], project_dir = get_dataset_path()
    dm = LocalDataManager(None)
    # get config
    cfg = load_config_data(project_dir + "/examples/urban_driver/config.yaml")

    # ===== INIT DATASET
    dataset_zarr = ChunkedDataset(dm.require(cfg[dataset_name]["key"])).open()

    vectorizer = build_vectorizer(cfg, dm)
    dataset_vec = EgoDatasetVectorized(cfg, dataset_zarr, vectorizer)
    print(dataset_vec)
    return dataset_zarr, dataset_vec, dm, cfg

if __name__ == "__main__":
    dataset_zarr, dataset_vec, dm, cfg = load_data(dataset_name="train_data_loader")
