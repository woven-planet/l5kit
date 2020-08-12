from collections import OrderedDict

import numpy as np

from . import read_gt_csv, read_pred_csv
from .metrics import neg_multi_log_likelihood


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

    def validate(ground_truth: dict, inference: dict) -> bool:
        valid = True

        if not (len(ground_truth.keys()) == len(inference.keys())):
            print(
                f"""Incorrect number of rows in inference csv. Expected {len(ground_truth.keys())},
                Got {len(inference.keys())}"""
            )
            valid = False

        missing_obstacles = ground_truth.keys() - inference.keys()
        if len(missing_obstacles):
            valid = False

        for missing_obstacle in missing_obstacles:
            print(f"Missing obstacle: {missing_obstacle}")

        unknown_obstacles = inference.keys() - ground_truth.keys()
        if len(unknown_obstacles):
            valid = False

        for unknown_obstacle in unknown_obstacles:
            print(f"Unknown obstacle: {unknown_obstacle}")

        return valid

    if not validate(ground_truth, inference):
        raise ValueError("Error validating csv, see above for details.")

    errors = []

    for key, ground_truth_value in ground_truth.items():
        gt_coord = ground_truth_value["coord"]
        avail = ground_truth_value["avail"]

        pred_coords = inference[key]["coords"]
        conf = inference[key]["conf"]
        errors.append(neg_multi_log_likelihood(gt_coord, pred_coords, conf, avail))
    return np.mean(errors)
