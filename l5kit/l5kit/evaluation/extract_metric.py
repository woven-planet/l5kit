from collections import OrderedDict

import numpy as np

from . import read_gt_csv, read_pred_csv
from .metrics import neg_multi_log_likelihood


def validate_dicts(ground_truth: dict, predicted: dict) -> bool:
    """
    Validate GT and pred dictionaries by comparing keys

    Args:
        ground_truth (dict): mapping from (track_id + timestamp) to an element returned from our csv utils
        predicted (dict): mapping from (track_id + timestamp) to an element returned from our csv utils

    Returns:
        (bool): True if the 2 dicts match (same keys)

    """
    valid = True

    num_agents_gt = len(ground_truth)
    num_agents_pred = len(predicted)

    if num_agents_gt != num_agents_pred:
        print(f"Incorrect number of rows in inference csv. Expected {num_agents_gt}, Got {num_agents_pred}")
        valid = False

    missing_agents = ground_truth.keys() - predicted.keys()
    if len(missing_agents):
        valid = False

    for missing_agents in missing_agents:
        print(f"Missing agents: {missing_agents}")

    unknown_agents = predicted.keys() - ground_truth.keys()
    if len(unknown_agents):
        valid = False

    for unknown_obstacle in unknown_agents:
        print(f"Unknown agents: {unknown_obstacle}")

    return valid


def compute_error_csv(ground_truth_path: str, inference_output_path: str) -> np.ndarray:
    """
    Arguments:
        ground_truth_path (str): Path to the ground truth csv file.
        inference_output_path (str): Path to the csv file containing network output.
    """
    ground_truth = OrderedDict()
    inference = OrderedDict()

    for el in read_gt_csv(ground_truth_path):
        ground_truth[el["track_id"] + el["timestamp"]] = el
    for el in read_pred_csv(inference_output_path):
        inference[el["track_id"] + el["timestamp"]] = el

    if not validate_dicts(ground_truth, inference):
        raise ValueError("Error validating csv, see above for details.")

    errors = []

    for key, ground_truth_value in ground_truth.items():
        gt_coord = ground_truth_value["coord"]
        avail = ground_truth_value["avail"]

        pred_coords = inference[key]["coords"]
        conf = inference[key]["conf"]
        errors.append(neg_multi_log_likelihood(gt_coord, pred_coords, conf, avail))
    return np.mean(errors)
