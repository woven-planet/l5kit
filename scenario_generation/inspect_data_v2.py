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
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer


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
    #  Get the  dataset
    ########################################################################
    zarr_dataset = ChunkedDataset(dm.require(cfg[dataset_name]["key"])).open()
    """
    ChunkedDataset is a dataset that lives on disk in compressed chunks, it has easy to use data loading and
    writing interfaces that involves making numpy-like slices.
    Currently only .zarr directory stores are supported (i.e. the data will live in a folder on your
    local filesystem called <something>.zarr).
    """

    vectorizer = build_vectorizer(cfg, dm)  # object that supports vectorization around an AV

    dataset_ego = EgoDatasetVectorized(cfg, zarr_dataset, vectorizer)
    """
    Get a PyTorch dataset object that can be used to train DNNs with vectorized input
    Args:
        cfg (dict): configuration file
        zarr_dataset (ChunkedDataset): the raw zarr dataset
        vectorizer (Vectorizer): a object that supports vectorization around an AV
        perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
    None if not desired
    """

    ########################################################################
    ## Setup the simulator class to be used to unroll the scene
    ########################################################################

    # ==== DEFINE CLOSED-LOOP SIMULATION
    num_simulation_steps = 20
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                               distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                               start_frame_index=0, show_info=True)

    scene_index = 3  # pick some scene

    sim_dataset = SimulationDataset.from_dataset_indices(dataset_vec, [scene_index], sim_cfg)
    """Create a SimulationDataset by picking indices from the provided dataset
    :param dataset: the EgoDataset
    :param scene_indices: scenes from the EgoDataset to pick
    :param sim_cfg: a simulation config
    :return: the new SimulationDataset
    """

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
