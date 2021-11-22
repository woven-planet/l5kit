import os

import torch

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.data import get_dataset_path
from l5kit.dataset import EgoDatasetVectorized
from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from l5kit.vectorization.vectorizer_builder import build_vectorizer


############################################################################################
def inspection(dataset_name="train_data_loader"):
    ########################################################################
    # Load data and configurations
    ########################################################################
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"], project_dir = get_dataset_path()

    dm = LocalDataManager(None)

    cfg = load_config_data(project_dir + "/examples/urban_driver/config.yaml")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ########################################################################
    #  Get the  dataset
    ########################################################################
    # ===== INIT DATASET
    eval_cfg = cfg["val_data_loader"]
    eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
    """
    ChunkedDataset is a dataset that lives on disk in compressed chunks, it has easy to use data loading and
    writing interfaces that involves making numpy-like slices.
    Currently only .zarr directory stores are supported (i.e. the data will live in a folder on your
    local filesystem called <something>.zarr).
    """
    vectorizer = build_vectorizer(cfg, dm)
    eval_dataset = EgoDatasetVectorized(cfg, eval_zarr, vectorizer)
    """
    Get a PyTorch dataset object that can be used to train DNNs with vectorized input
    Args:
        cfg (dict): configuration file
        zarr_dataset (ChunkedDataset): the raw zarr dataset
        vectorizer (Vectorizer): a object that supports vectorization around an AV
        perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
    None if not desired
    """
    print(eval_dataset)



    ########################################################################
    ## Setup the simulator class to be used to unroll the scene
    ########################################################################

    # ==== DEFINE CLOSED-LOOP SIMULATION
    num_simulation_steps = 50

    use_agents_gt = False

    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=use_agents_gt, disable_new_agents=True,
                               distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                               start_frame_index=0, show_info=True)
    """ Defines the parameters used for the simulation of ego and agents around it.

    :param use_ego_gt: whether to use GT annotations for ego instead of model's outputs
    :param use_agents_gt: whether to use GT annotations for agents instead of model's outputs
    :param disable_new_agents: whether to disable agents that are not returned at start_frame_index
    :param distance_th_far: if a tracked agent is closed than this value to ego, it will be controlled
    :param distance_th_close: if a new agent is closer than this value to ego, it will be controlled
    :param start_frame_index: the start index of the simulation
    :param num_simulation_steps: the number of step to simulate
    :param show_info: whether to show info logging during unroll
    """

    model_path = project_dir + "/urban_driver_dummy_model.pt"
    model_ego = torch.load(model_path).to(device)
    model_ego = model_ego.eval()

    model_path = project_dir + "/urban_driver_dummy_model.pt"
    model_agents = torch.load(model_path).to(device)
    model_agents = model_agents.eval()



    sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model_ego, model_agents=model_agents)
    """
       Create a simulation loop object capable of unrolling ego and agents
       :param sim_cfg: configuration for unroll
       :param dataset: EgoDataset used while unrolling
       :param device: a torch device. Inference will be performed here
       :param model_ego: the model to be used for ego
       :param model_agents: the model to be used for agents
       :param keys_to_exclude: keys to exclude from input/output (e.g. huge blobs)
       :param mode: the framework that uses the closed loop simulator
   """

    # scenes from the EgoDataset to pick
    scene_indices = [0]

    torch.set_grad_enabled(False) ########## The unroll gives an error if this is not used-
    # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    # but we can do backprop in time if this is used
    # we should probably use VectorizedUnrollModel instead (as in urban_driver/train )

    simulated_outputs = sim_loop.unroll(scene_indices)
    """
    Simulate the dataset for the given scene indices
    :param scene_indices: the scene indices we want to simulate
    :return: the simulated dataset
    """


    ########################################################################
    #  Transform back from vectorized representation
    ########################################################################

    ########################################################################
    #  Plot initial scene
    ########################################################################

    return None


############################################################################################


if __name__ == "__main__":
    inspection(dataset_name="train_data_loader")
