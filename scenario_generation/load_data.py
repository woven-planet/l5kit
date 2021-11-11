
import os

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset,get_frames_slice_from_scenes
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.data import get_dataset_path
from l5kit.sampling.agent_sampling_vectorized import generate_agent_sample_vectorized


############################################################################################
def load_data(dataset_name="train_data_loader"):


    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"], project_dir = get_dataset_path()

    dm = LocalDataManager(None)

    cfg = load_config_data(project_dir + "/examples/urban_driver/config.yaml")

    #   the raw zarr format dataset  (object to sample data from)
    zarr_dataset = ChunkedDataset(dm.require(cfg[dataset_name]["key"])).open()

    # object that supports vectorization around an AV
    vectorizer = build_vectorizer(cfg, dm)

    #  dataset  in vector format (object to sample data from)
    dataset_vec = EgoDatasetVectorized(cfg, zarr_dataset, vectorizer)

    # test the vectorization function
    frames = zarr_dataset.frames[get_frames_slice_from_scenes(zarr_dataset.scenes[0])]
    sampled_data = generate_agent_sample_vectorized(0, frames, zarr_dataset.agents, zarr_dataset.tl_faces, None,
                                            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
                                            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
                                            future_num_frames=cfg["model_params"]["future_num_frames"],
                                            step_time=cfg["model_params"]["step_time"],
                                            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
                                            vectorizer=build_vectorizer(cfg, dm))


    return zarr_dataset, dataset_vec, dm, cfg
############################################################################################


if __name__ == "__main__":
    zarr_dataset, dataset_vec, dm, cfg = load_data(dataset_name="train_data_loader")
