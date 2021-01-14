import csv
from itertools import chain
from typing import Iterator, List, Optional

import numpy as np


MAX_MODES = 3

"""
These utilities write and read csv with ground-truth and prediction data.
Both share the first two fields (timestamp and track_id) which are used to identify a single record.

GT:
Single mode future prediction with availabilities (either 1->available or 0->unavailable).
Header fields have these meanings:
timestamp, track_id, avail_time_0, avail_time_1, ..., coord_x_time_0, coord_y_time_0, ...

PRED:
Multi mode future prediction with confidence score (one value per mode up to MAX_MODES, sum to 1).
Header fields have these meanings:
timestamp, track_id, conf_mode_0, conf_mode_1, ..., coord_x_time_0_mode_0, coord_y_time_0_mode_0, ...,
coord_x_time_0_mode_1, coord_y_time_0_mode_1, ...

"""


def _generate_coords_keys(future_len: int, mode_index: int = 0) -> List[str]:
    """
    Generate keys like coord_x00, coord_y00... that can be used to get or set value in CSV.
    Two keys for each mode and future step.

    Args:
        future_len (int): how many prediction the data has in the future
        mode_index (int): what mode are we reading/writing

    Returns:
        List[str]: a list of keys
    """
    return list(
        chain.from_iterable([[f"coord_x{mode_index}{i}", f"coord_y{mode_index}{i}"] for i in range(future_len)])
    )


def _generate_avails_keys(future_len: int) -> List[str]:
    """
    Generate availabilities keys (one per future step)

    Args:
        future_len (int): how many prediction in the future data has

    Returns:
        List[str]: a list of keys
    """
    return [f"avail_{i}" for i in range(future_len)]


def _generate_confs_keys() -> List[str]:
    """
    Generate modes keys (one per mode)

    Returns:
        List[str]: a list of keys
    """
    return [f"conf_{i}" for i in range(MAX_MODES)]


def write_gt_csv(
        csv_path: str, timestamps: np.ndarray, track_ids: np.ndarray, coords: np.ndarray, avails: np.ndarray
) -> None:
    """
    Encode the ground truth into a csv file

    Args:
        csv_path (str): path to the csv to write
        timestamps (np.ndarray): (num_example,) frame timestamps
        track_ids (np.ndarray): (num_example,) agent ids
        coords (np.ndarray): (num_example x future_len x num_coords) meters displacements
        avails (np.ndarray): (num_example x future_len) array with value 0 (discard in eval) or 1 (keep in eval)

    Returns:

    """
    assert len(coords.shape) == 3
    num_example, future_len, num_coords = coords.shape
    assert num_coords == 2
    assert timestamps.shape == track_ids.shape == (num_example,)
    assert avails.shape == (num_example, future_len)

    coords_keys = _generate_coords_keys(future_len)
    avails_keys = _generate_avails_keys(future_len)

    # create and write HEADER
    # order is (timestamp,track_id,availabilities,coords)
    fieldnames = ["timestamp", "track_id"] + avails_keys + coords_keys
    writer = csv.DictWriter(open(csv_path, "w"), fieldnames)
    writer.writeheader()

    for timestamp, track_id, coord, avail in zip(timestamps, track_ids, coords, avails):
        # writing using dict relieves us from respecting the order
        line = {"timestamp": timestamp, "track_id": track_id}
        line.update({key: ava for key, ava in zip(avails_keys, avail)})
        line.update({key: f"{cor:.5f}" for key, cor in zip(coords_keys, coord.reshape(-1))})

        writer.writerow(line)


def read_gt_csv(csv_path: str) -> Iterator[dict]:
    """
    Generator function that returns a line at a time from the csv file as a dict

    Args:
        csv_path (str): path of the csv to read

    Returns:
        Iterator[dict]: dict keys are the csv header fieldnames
    """
    reader = csv.DictReader(open(csv_path, "r"))
    fieldnames = reader.fieldnames
    assert fieldnames is not None, "error reading fieldnames"

    future_len = (len(fieldnames) - 2) / 3  # excluded first two fields, the rest must be (x, y, av) * len = 3*len
    assert future_len == int(future_len), "error estimating len"
    future_len = int(future_len)

    coords_keys = _generate_coords_keys(future_len)
    avails_keys = _generate_avails_keys(future_len)

    for row in reader:
        timestamp = row["timestamp"]
        track_id = row["track_id"]

        avail = np.asarray([np.float64(row[key]) for key in avails_keys])

        coord = np.asarray([np.float64(row[key]) for key in coords_keys])
        coord = coord.reshape((future_len, 2))

        yield {"track_id": track_id, "timestamp": timestamp, "coord": coord, "avail": avail}


def write_pred_csv(
        csv_path: str,
        timestamps: np.ndarray,
        track_ids: np.ndarray,
        coords: np.ndarray,
        confs: Optional[np.ndarray] = None,
) -> None:
    """
    Encode the predictions into a csv file. Coords can have an additional axis for multi-mode.
    We handle up to MAX_MODES modes.
    For the uni-modal case (i.e. all predictions have just a single mode), coords should not have the additional axis
    and confs should be set to None. In this case, a single mode with confidence 1 will be written.

    Args:
        csv_path (str): path to the csv to write
        timestamps (np.ndarray): (num_example,) frame timestamps
        track_ids (np.ndarray): (num_example,) agent ids
        coords (np.ndarray): (num_example x (modes) x future_len x num_coords) meters displacements
        confs (Optional[np.ndarray]): (num_example x modes) confidence of each modes in each example.
        Rows should sum to 1

    Returns:

    """
    assert len(coords.shape) in [3, 4]

    if len(coords.shape) == 3:
        assert confs is None  # no conf for the single-mode case
        coords = np.expand_dims(coords, 1)  # add a new axis for the multi-mode
        confs = np.ones((len(coords), 1))  # full confidence

    num_example, num_modes, future_len, num_coords = coords.shape
    assert num_coords == 2
    assert timestamps.shape == track_ids.shape == (num_example,)
    assert confs is not None and confs.shape == (num_example, num_modes)
    assert np.allclose(np.sum(confs, axis=-1), 1.0)
    assert num_modes <= MAX_MODES

    # generate always a fixed size json for MAX_MODES by padding the arrays with zeros
    coords_padded = np.zeros((num_example, MAX_MODES, future_len, num_coords), dtype=coords.dtype)
    coords_padded[:, :num_modes] = coords
    confs_padded = np.zeros((num_example, MAX_MODES), dtype=confs.dtype)
    confs_padded[:, :num_modes] = confs

    coords_keys_list = [_generate_coords_keys(future_len, mode_index=idx) for idx in range(MAX_MODES)]
    confs_keys = _generate_confs_keys()

    # create and write HEADER
    # order is (timestamp,track_id,confidences,coords)
    fieldnames = ["timestamp", "track_id"] + confs_keys  # all confidences before coordinates
    for coords_labels in coords_keys_list:
        fieldnames.extend(coords_labels)

    writer = csv.DictWriter(open(csv_path, "w"), fieldnames)
    writer.writeheader()

    for timestamp, track_id, coord, conf in zip(timestamps, track_ids, coords_padded, confs_padded):
        line = {"timestamp": timestamp, "track_id": track_id}
        line.update({key: con for key, con in zip(confs_keys, conf)})

        for idx in range(MAX_MODES):
            line.update({key: f"{cor:.5f}" for key, cor in zip(coords_keys_list[idx], coord[idx].reshape(-1))})

        writer.writerow(line)


def read_pred_csv(csv_path: str) -> Iterator[dict]:
    """
    Generator function that returns a line at the time from the csv file as a dict

    Args:
        csv_path (str): path of the csv to read

    Returns:
        Iterator[dict]: dict keys are the csv header fieldnames
    """

    reader = csv.DictReader(open(csv_path, "r"))
    fieldnames = reader.fieldnames
    assert fieldnames is not None, "error reading fieldnames"

    # exclude timestamp, track_id and MAX_MODES confs, the rest should be (x, y) * len * 3 = 6*len
    future_len = (len(fieldnames) - (2 + MAX_MODES)) / 6
    assert future_len == int(future_len), "error estimating len"
    future_len = int(future_len)

    coords_labels_list = [_generate_coords_keys(future_len, mode_index=idx) for idx in range(MAX_MODES)]
    confs_labels = _generate_confs_keys()

    for row in reader:
        track_id = row["track_id"]
        timestamp = row["timestamp"]

        conf = np.asarray([np.float64(row[conf_label]) for conf_label in confs_labels])

        coords = []
        for idx in range(MAX_MODES):
            coord = np.asarray([np.float64(row[coord_label]) for coord_label in coords_labels_list[idx]])
            coords.append(coord.reshape((future_len, 2)))

        coords = np.stack(coords, axis=0)

        yield {"track_id": track_id, "timestamp": timestamp, "coords": coords, "conf": conf}
