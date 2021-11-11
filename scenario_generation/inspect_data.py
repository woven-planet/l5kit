import os

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.data import get_dataset_path
from l5kit.sampling.agent_sampling_vectorized import generate_agent_sample_vectorized


############################################################################################
def load_data(dataset_name="train_data_loader"):
    ########################################################################
    # Load data and configurations
    ########################################################################
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"], project_dir = get_dataset_path()

    dm = LocalDataManager(None)

    cfg = load_config_data(project_dir + "/examples/urban_driver/config.yaml")

    ########################################################################
    #  Get the raw  dataset object (in zarr format)
    ########################################################################
    zarr_dataset = ChunkedDataset(dm.require(cfg[dataset_name]["key"])).open()
    """ChunkedDataset is a dataset that lives on disk in compressed chunks, it has easy to use data loading and
    writing interfaces that involves making numpy-like slices.
    Currently only .zarr directory stores are supported (i.e. the data will live in a folder on your
    local filesystem called <something>.zarr).
    """

    ########################################################################
    # Take a single scene
    ########################################################################
    """Get a new ChunkedDataset of a single scene.
    This dataset lives in memory (as np.ndarray)

    :param scene_index: the scene index
    :return: a dataset with a single scene inside
    """
    scene_index = 3
    zarr_dataset = zarr_dataset.get_scene_dataset(scene_index)


    ########################################################################
    # Transform to vectorized form and extract only initial time-step (t=0)
    ########################################################################
    # object that supports vectorization around an AV
    vectorizer = build_vectorizer(cfg, dm)
    """
    Get a PyTorch dataset object that can be used to train DNNs with vectorized input
    Args:
        cfg (dict): configuration file
        zarr_dataset (ChunkedDataset): the raw zarr dataset
        vectorizer (Vectorizer): a object that supports vectorization around an AV
        perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
    None if not desired
    """
    dataset_vec = EgoDatasetVectorized(cfg, zarr_dataset, vectorizer)

    # test the vectorization function
    scene_index = 31
    frames = zarr_dataset.frames[get_frames_slice_from_scenes(zarr_dataset.scenes[0])]
    timestep_index = 0
    sampled_data = generate_agent_sample_vectorized(timestep_index, frames, zarr_dataset.agents, zarr_dataset.tl_faces, None,
                                                    history_num_frames_ego=cfg["model_params"][
                                                        "history_num_frames_ego"],
                                                    history_num_frames_agents=cfg["model_params"][
                                                        "history_num_frames_agents"],
                                                    future_num_frames=cfg["model_params"]["future_num_frames"],
                                                    step_time=cfg["model_params"]["step_time"],
                                                    filter_agents_threshold=cfg["raster_params"][
                                                        "filter_agents_threshold"],
                                                    vectorizer=build_vectorizer(cfg, dm))
    """
    Generates the inputs and targets to train a deep prediction model with vectorized inputs.
    A deep prediction model takes as input the state of the world in vectorized form,
    and outputs where that agent will be some seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the representation and the returned targets are derived from
        their future states.
        history_num_frames_ego (int): Amount of ego history frames to include
        history_num_frames_agents (int): Amount of agent history frames to include
        future_num_frames (int): Amount of future frames to include
        step_time (float): seconds between consecutive steps
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
        to train models that can recover from slight divergence from training set data    
        
        Returns:
        dict: a dict containing e.g. the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask,
        the vectorized input representation features, and (optional) a raster image
    """



    ########################################################################
    #  Plot initial scene
    ########################################################################

    return zarr_dataset, dataset_vec, dm, cfg


############################################################################################


if __name__ == "__main__":
    zarr_dataset, dataset_vec, dm, cfg = load_data(dataset_name="train_data_loader")
