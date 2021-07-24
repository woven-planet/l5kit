""" Utils for Closed Loop Evaluation """

from prettytable import PrettyTable

from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator
from l5kit.cle.validators import ValidationCountingAggregator


def aggregate_cle_metrics(cle_evaluator: ClosedLoopEvaluator) -> None:
    validation_results_log = cle_evaluator.validation_results()
    agg_log = ValidationCountingAggregator().aggregate(validation_results_log)
    cle_evaluator.reset()

    fields = ["metric", "log_replayed agents"]
    table = PrettyTable(field_names=fields)
    for metric_name in agg_log:
        table.add_row([metric_name, agg_log[metric_name].item()])
    print(table)

from typing import DefaultDict, List

from l5kit.simulation.dataset import SimulationDataset
from l5kit.simulation.unroll import SimulationOutput, UnrollInputOutput

class SimulationOutputGym(SimulationOutput):
    def __init__(self, scene_id: int, sim_dataset: SimulationDataset,
                 ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]],
                 agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]]):
        """This object holds information about the result of the simulation loop
        for a given scene dataset in Gym
        :param scene_id: the scene indices
        :param sim_dataset: the simulation dataset
        :param ego_ins_outs: all inputs and outputs for ego (each frame of each scene has only one)
        :param agents_ins_outs: all inputs and outputs for agents (multiple per frame in a scene)
        """
        super(SimulationOutputGym, self).__init__(scene_id, sim_dataset, ego_ins_outs, agents_ins_outs)

        # Required for Bokeh Visualizer
        self.tls_frames = self.simulated_dataset.dataset.tl_faces
        self.agents_th = self.simulated_dataset.cfg["raster_params"]["filter_agents_threshold"]

        # Remove Dataset attributes
        del self.recorded_dataset
        del self.simulated_dataset