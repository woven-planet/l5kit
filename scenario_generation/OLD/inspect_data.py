import os
import torch
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.data import get_dataset_path
from l5kit.sampling.agent_sampling_vectorized import generate_agent_sample_vectorized
from torch.utils.data.dataloader import default_collate
from l5kit.dataset.utils import move_to_device, move_to_numpy
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import ClosedLoopSimulator

############################################################################################
def inspection(dataset_name="train_data_loader"):
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
    """
    ###########################################
    #### ChunkedDataset ####
    ###########################################
    ChunkedDataset is a dataset that lives on disk in compressed chunks, it has easy to use data loading and
    writing interfaces that involves making numpy-like slices.
    Currently only .zarr directory stores are supported (i.e. the data will live in a folder on your
    local filesystem called <something>.zarr).
    """

    ########################################################################
    # Take a single scene
    ########################################################################
    """
    ###########################################
    #### get_scene_dataset ####
    ###########################################
    Get a new ChunkedDataset of a single scene.
    This dataset lives in memory (as np.ndarray)

    :param scene_index: the scene index
    :return: a dataset with a single scene inside
    """
    scene_index = 3
    zarr_dataset = zarr_dataset.get_scene_dataset(scene_index)



    ########################################################################
    # Transform to vectorized
    ########################################################################
    # object that supports vectorization around an AV
    vectorizer = build_vectorizer(cfg, dm)
    """
    ###########################################
    #### build_vectorizer ####
    ###########################################
    Get a PyTorch dataset object that can be used to train DNNs with vectorized input
    Args:
        cfg (dict): configuration file
        zarr_dataset (ChunkedDataset): the raw zarr dataset
        vectorizer (Vectorizer): a object that supports vectorization around an AV
        perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
    None if not desired
    """
    dataset_vec = EgoDatasetVectorized(cfg, zarr_dataset, vectorizer)


    frames = zarr_dataset.frames[get_frames_slice_from_scenes(zarr_dataset.scenes[0])]


    ########################################################################
    # extract only initial time-step (t=0)
    ########################################################################
    timestep_index = 0
    sampled_data = generate_agent_sample_vectorized(timestep_index, frames, zarr_dataset.agents, zarr_dataset.tl_faces, None,
                                                    history_num_frames_ego=1,  # what should we use?
                                                    history_num_frames_agents=1,
                                                    future_num_frames=1,  # we must take at least 1 to compute velocity
                                                    step_time=cfg["model_params"]["step_time"],
                                                    filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
                                                    vectorizer=build_vectorizer(cfg, dm))
    """
    ###########################################
    #### generate_agent_sample_vectorized ####
    ###########################################
    Generates the inputs and targets to train a deep prediction model with vectorized inputs.
    A deep prediction model takes as input the state of the world in vectorized form,
    and outputs where that agent will be some seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        
        TODO: what are the exact variables and dimensions?
        
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
        
        Note that the vectorized features are part of the output dict:
            vectorized_features = vectorizer.vectorize(selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world,
                                               history_frames, history_agents, history_tl_faces, history_coords_offset,
                                               history_yaws_offset, history_availability, future_frames, future_agents)
           return {**frame_info, **vectorized_features}
    """

    """
    ###########################################
    #### vectorize ####
    ###########################################
    Base function to execute a vectorization process.

    Arguments:
        selected_track_id: selected_track_id: Either None for AV, or the ID of an agent that you want to
        predict the future of.
        This agent is centered in the representation and the returned targets are derived from their future states.
        agent_centroid_m: position of the target agent
        agent_yaw_rad: yaw angle of the target agent
        agent_from_world: inverted agent pose as 3x3 matrix
        history_frames: historical frames of the target frame
        history_agents: agents appearing in history_frames
        history_tl_faces: traffic light faces in history frames
        history_position_m: historical positions of target agent
        history_yaws_rad: historical yaws of target agent
        history_availability: availability mask of history frames
        future_frames: future frames of the target frame
        future_agents: agents in future_frames
        
    ["history_positions"].shape == (max_history_num_frames + 1, 2)
    ["history_yaws"].shape == (max_history_num_frames + 1, 1)
    ["history_extents"].shape == (max_history_num_frames + 1, 2)
    ["history_availabilities"].shape == (max_history_num_frames + 1,)

    ["all_other_agents_history_positions"].shape == (num_agents, max_history_num_frames + 1, 2)
     # num_other_agents (M) x sequence_length x 2 (two being x, y)
    ["all_other_agents_history_yaws"].shape == (num_agents, max_history_num_frames + 1, 1) 
    # num_other_agents (M) x sequence_length x 1
    ["all_other_agents_history_extents"].shape == (num_agents, max_history_num_frames + 1, 2) 
     # agent_extent = (EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH)
    ["all_other_agents_history_availability"].shape == (num_agents, max_history_num_frames + 1,)

    ["target_positions"].shape == (future_num_frames, 2)
    ["target_yaws"].shape == (future_num_frames, 1)
    ["target_extents"].shape == (future_num_frames, 2)
    ["target_availabilities"].shape == (future_num_frames,)

    ["all_other_agents_future_positions"].shape == (num_agents, future_num_frames, 2) 
    ["all_other_agents_future_yaws"].shape == (num_agents, future_num_frames, 1)
    ["all_other_agents_future_extents"].shape == (num_agents, future_num_frames, 2)  # agent_extent = (EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH)
    ["all_other_agents_future_availability"].shape == (num_agents, future_num_frames,)
    ["all_other_agents_types"].shape == (num_agents,)

    ["agent_trajectory_polyline"].shape == (max_history_num_frames + 1, 3)
    ["agent_polyline_availability"].shape == (max_history_num_frames + 1,)
    ["other_agents_polyline"].shape == (num_agents, max_history_num_frames + 1, 3)
    ["other_agents_polyline_availability"].shape == (num_agents, max_history_num_frames + 1,)

    Returns:
        dict: a dict containing the vectorized frame representation
    """

    ####################################################################################

    ########################################################################
    ## Setup the simulator class to be used to unroll the scene
    ########################################################################
    scene_indices = [0]

    # ==== DEFINE CLOSED-LOOP SIMULATION
    num_simulation_steps = 1
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                               distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                               start_frame_index=0, show_info=True)

    sim_dataset = SimulationDataset.from_dataset_indices(dataset_vec, scene_indices, sim_cfg)

    ########################################################################
    ## Get next state using motion prediction model
    ########################################################################
    """   
      VectorizedUnrollModel(VectorizedModel): Vectorized closed-loop planning model.
    """
    model_path = project_dir + "/urban_driver_dummy_model.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path).to(device)
    model = model.eval()

    ego_input = [dataset_vec[0]]  # list of a batch of states, here we used only one state
    # a state is a dict[str:tensor] with 38 fields

    ego_input = default_collate(ego_input)  # This concatenates the tensors at each field

    ego_input = move_to_device(ego_input, device)


    # Get prediction output for ego:
    # 'positions' : torch.Size([batch_size, future_num_frames, 2]
    # 'yaws' :  torch.Size([batch_size, future_num_frames, 1]
    ego_output = model(ego_input)
    pass
    print(ego_output)

    agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)

    next_frame_index = 1
    # ClosedLoopSimulator.update_agents(sim_dataset, next_frame_index, agents_input_dict, agents_output_dict)
    ClosedLoopSimulator.update_ego(sim_dataset, next_frame_index, (ego_input), (ego_output))
    ########################################################################
    #  Transform back from vectorized representation
    ########################################################################

    ########################################################################
    #  Plot initial scene
    ########################################################################




    return zarr_dataset, dataset_vec, dm, cfg


############################################################################################


if __name__ == "__main__":
    zarr_dataset, dataset_vec, dm, cfg = inspection(dataset_name="train_data_loader")
