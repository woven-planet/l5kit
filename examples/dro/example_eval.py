# Example Evaluation

from drivenet_eval import eval_model
# Give model path and make sure config.yaml respects the model
model_path = "./checkpoints/drivenet_h0_p05_onecycle_schedule_step5_vrex_50epochs_1262400_steps.pt"
scene_id_to_type_path = '../../dataset_metadata/validate_turns_metadata.csv'

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from stable_baselines3.common import utils
import torch

dm = LocalDataManager(None)
# get config
cfg = load_config_data("./drivenet_config.yaml")
# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)

# Validation Dataset
eval_cfg = cfg["val_data_loader"]
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
# For evaluation
num_scenes_to_unroll = eval_cfg["max_scene_id"]

logger = utils.configure_logger(1, "./drivenet_logs/", "eval", True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path).to(device)
model = model.eval()

import time
st = time.time()
# eval_model(model, eval_dataset, logger, "eval", 2000000, num_scenes_to_unroll, num_simulation_steps=None)
eval_model(model, eval_dataset, logger, "eval", 2000000, num_scenes_to_unroll=4000, num_simulation_steps=None,
           enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_path)
print("Time: ", time.time() - st)
