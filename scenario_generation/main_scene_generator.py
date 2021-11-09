
import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer

from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator

from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI

from collections import defaultdict
import os


# set env variable for data
root_project_dir = os.path.dirname(os.getcwd())
os.environ["L5KIT_DATA_FOLDER"] = open(root_project_dir + "/examples/dataset_dir.txt", "r").read().strip()
dm = LocalDataManager(None)
# get config
cfg = load_config_data(root_project_dir + "/examples/urban_driver/config.yaml")

## Load the model

model_path = "/tmp/urban_driver.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path).to(device)
model = model.eval()
torch.set_grad_enabled(False)

## Load the evaluation data
# Differently from training and open loop evaluation, this setting is intrinsically sequential. As such, we won't be using any of PyTorch's parallelisation functionalities.


# ===== INIT DATASET
eval_cfg = cfg["val_data_loader"]
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
vectorizer = build_vectorizer(cfg, dm)
eval_dataset = EgoDatasetVectorized(cfg, eval_zarr, vectorizer)
print(eval_dataset)

## Define some simulation properties
# We define here some common simulation properties such as the length of the simulation and how many scene to simulate.
# **NOTE: these properties have a significant impact on the execution time. We suggest you to increase them only if your setup includes a GPU.**

num_scenes_to_unroll = 10
num_simulation_steps = 50

# Closed-loop simulation

# We define a closed-loop simulation that drives the SDV for `num_simulation_steps` steps while using the log-replayed agents.
# Then, we unroll the selected scenes.
# The simulation output contains all the information related to the scene, including the annotated and simulated positions, states, and trajectories of the SDV and the agents.
# If you want to know more about what the simulation output contains, please refer to the source code of the class `SimulationOutput`.

# ==== DEFINE CLOSED-LOOP SIMULATION
sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                           distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                           start_frame_index=0, show_info=True)

sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model, model_agents=None)

# ==== UNROLL
scenes_to_unroll = list(range(0, len(eval_zarr.scenes), len(eval_zarr.scenes)//num_scenes_to_unroll))
sim_outs = sim_loop.unroll(scenes_to_unroll)

# Closed-loop metrics
# **Note: for a detailed explanation of CLE metrics, please refer again to our [planning notebook](../planning/closed_loop_test.ipynb)**

metrics = [DisplacementErrorL2Metric(),
           DistanceToRefTrajectoryMetric(),
           CollisionFrontMetric(),
           CollisionRearMetric(),
           CollisionSideMetric()]

validators = [RangeValidator("displacement_error_l2", DisplacementErrorL2Metric, max_value=30),
              RangeValidator("distance_ref_trajectory", DistanceToRefTrajectoryMetric, max_value=4),
              RangeValidator("collision_front", CollisionFrontMetric, max_value=0),
              RangeValidator("collision_rear", CollisionRearMetric, max_value=0),
              RangeValidator("collision_side", CollisionSideMetric, max_value=0)]

intervention_validators = ["displacement_error_l2",
                           "distance_ref_trajectory",
                           "collision_front",
                           "collision_rear",
                           "collision_side"]

cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
                                                   validators=validators,
                                                   composite_metrics=[],
                                                   intervention_validators=intervention_validators))

#%% md

# Quantitative evaluation
# We can now compute the metric evaluation, collect the results and aggregate them.
#%%
cle_evaluator.evaluate(sim_outs)
validation_results = cle_evaluator.validation_results()
agg = ValidationCountingAggregator().aggregate(validation_results)
cle_evaluator.reset()


## Reporting errors from the closed-loop
fields = ["metric", "value"]
table = PrettyTable(field_names=fields)

values = []
names = []

for metric_name in agg:
    table.add_row([metric_name, agg[metric_name].item()])
    values.append(agg[metric_name].item())
    names.append(metric_name)

print(table)
plt.bar(np.arange(len(names)), values)
plt.xticks(np.arange(len(names)), names, rotation=60, ha='right')
plt.show()

# Qualitative evaluation

## Visualise the closed-loop
# We can visualise the scenes we have obtained previously.
# **The policy is now in full control of the SDV as this moves through the annotated scene.**

output_notebook()
mapAPI = MapAPI.from_cfg(dm, cfg)
for sim_out in sim_outs: # for each scene
    vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
    show(visualize(sim_out.scene_id, vis_in))


