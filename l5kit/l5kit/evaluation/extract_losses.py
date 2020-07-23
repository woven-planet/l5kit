import argparse
import csv
from collections import OrderedDict

import numpy as np

from .metrics import single_trajectory_metric


def compute_mse_error_csv(ground_truth_path: str, inference_output_path: str) -> np.ndarray:
    """
    Arguments:
        ground_truth_path (str): Path to the ground truth csv file.
        inference_output_path (str): Path to the csv file containing network output.
    """

    def extract_csv(csv_path: str) -> list:
        with open(csv_path, "r", newline="") as csv_file:
            return [row for row in csv.reader(csv_file, delimiter=",")]

    ground_truth_rows = extract_csv(ground_truth_path)[1:]
    inference_rows = extract_csv(inference_output_path)[1:]

    def to_key(vals: str) -> str:
        return str(",".join(vals))

    def parse_values(values: list) -> np.ndarray:
        return np.reshape(np.array(values).astype(np.float64), (-1, 2))

    ground_truth = OrderedDict()
    inference = OrderedDict()
    for gt_row in ground_truth_rows:
        ground_truth[to_key(gt_row[:2])] = parse_values(gt_row[2:])
    for inf_row in inference_rows:
        inference[to_key(inf_row[:2])] = parse_values(inf_row[2:])

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

    ground_truth_values = np.stack(list(ground_truth.values()))
    inference_values = np.stack(list(inference.values()))
    return single_trajectory_metric(ground_truth_values, inference_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Print mean squared error for a deep prediction run. Takes as input a csv file for ground truth,
                                     another as output from inference. """
    )
    parser.add_argument("ground_truth_csv", type=str, help="Path to the csv containing ground truth.")
    parser.add_argument("inference_csv", type=str, help="Path to the csv containing output from an inference run.")
    args = parser.parse_args()

    mse = compute_mse_error_csv(args.ground_truth_csv, args.inference_csv)
    print("mse", mse)
