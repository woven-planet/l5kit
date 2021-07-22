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
