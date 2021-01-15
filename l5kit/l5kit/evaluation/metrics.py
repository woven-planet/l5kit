from typing import Callable, Tuple, Optional
from enum import IntEnum

import numpy as np
from shapely.geometry import LineString, Polygon


metric_signature = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _assert_shapes(ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray) -> None:
    """
    Check the shapes of args required by metrics

    Args:
        ground_truth (np.ndarray): array of shape (timesteps)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (timesteps) with the availability for each gt timesteps

    Returns:

    """
    assert len(pred.shape) == 3, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    num_modes, future_len, num_coords = pred.shape

    assert ground_truth.shape == (
        future_len,
        num_coords,
    ), f"expected 2D (Time x Coords) array for gt, got {ground_truth.shape}"
    assert confidences.shape == (num_modes,), f"expected 1D (Modes) array for confidences, got {confidences.shape}"
    assert np.allclose(np.sum(confidences), 1), "confidences should sum to 1"
    assert avails.shape == (future_len,), f"expected 1D (Time) array for avails, got {avails.shape}"
    # assert all data are valid
    assert np.isfinite(pred).all(), "invalid value found in pred"
    assert np.isfinite(ground_truth).all(), "invalid value found in gt"
    assert np.isfinite(confidences).all(), "invalid value found in confidences"
    assert np.isfinite(avails).all(), "invalid value found in avails"


def neg_multi_log_likelihood(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray
) -> np.ndarray:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/

    For more details about used loss function and reformulation, please see
    https://github.com/lyft/l5kit/blob/master/competition.md.

    Args:
        ground_truth (np.ndarray): array of shape (timesteps)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (timesteps) with the availability for each gt timesteps

    Returns:
        np.ndarray: negative log-likelihood for this example, a single float number
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce timesteps

    # use max aggregator on modes for numerical stability
    max_value = error.max()  # error are negative at this point, so max() gives the minimum one
    error = -np.log(np.sum(np.exp(error - max_value), axis=-1)) - max_value  # reduce modes
    return error


def rmse(ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray) -> np.ndarray:
    """
    Return the root mean squared error, computed using the stable nll

    Args:
        ground_truth (np.ndarray): array of shape (timesteps)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (timesteps) with the availability for each gt timesteps

    Returns:
        np.ndarray: negative log-likelihood for this example, a single float number

    """
    nll = neg_multi_log_likelihood(ground_truth, pred, confidences, avails)
    _, future_len, _ = pred.shape

    return np.sqrt(2 * nll / future_len)


def prob_true_mode(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray
) -> np.ndarray:
    """
    Return the probability of the true mode

    Args:
        ground_truth (np.ndarray): array of shape (timesteps)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (timesteps) with the availability for each gt timesteps

    Returns:
        np.ndarray: a (modes) numpy array

    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce timesteps

    # use max aggregator on modes for numerical stability
    max_value = error.max()  # error are negative at this point, so max() gives the minimum one

    error = np.exp(error - max_value) / np.sum(np.exp(error - max_value))
    return error


def time_displace(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray
) -> np.ndarray:
    """
    Return the displacement at timesteps T

    Args:
        ground_truth (np.ndarray): array of shape (timesteps)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (timesteps) with the availability for each gt timesteps

    Returns:
        np.ndarray: a (timesteps) numpy array

    """
    true_mode_error = prob_true_mode(ground_truth, pred, confidences, avails)
    true_mode_error = true_mode_error[:, None]  # add timesteps axis

    ground_truth = np.expand_dims(ground_truth, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    return np.sum(true_mode_error * np.sqrt(error), axis=0)  # reduce modes


def _average_displacement_error(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray, mode: str
) -> np.ndarray:
    """
    Returns the average displacement error (ADE), which is the average displacement over all timesteps.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses

    Args:
        ground_truth (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)

    Returns:
        np.ndarray: average displacement error (ADE), a single float number
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = np.mean(error, axis=-1)  # average over timesteps

    if mode == "oracle":
        error = np.min(error)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=0)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error


def average_displacement_error_oracle(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray,
) -> np.ndarray:
    """
    Calls _average_displacement_error() to get the oracle average displacement error.

    Args:
        ground_truth (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: oracle average displacement error (ADE), a single float number
    """

    return _average_displacement_error(ground_truth, pred, confidences, avails, "oracle")


def average_displacement_error_mean(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray,
) -> np.ndarray:
    """
    Calls _average_displacement_error() to get the mean average displacement error.

    Args:
        ground_truth (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: mean average displacement error (ADE), a single float number
    """

    return _average_displacement_error(ground_truth, pred, confidences, avails, "mean")


def _final_displacement_error(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray, mode: str
) -> np.ndarray:
    """
    Returns the final displacement error (FDE), which is the displacement calculated at the last timestep.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses

    Args:
        ground_truth (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)

    Returns:
        np.ndarray: final displacement error (FDE), a single float number
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = error[:, -1]  # use last timestep

    if mode == "oracle":
        error = np.min(error)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=0)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error


def final_displacement_error_oracle(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray,
) -> np.ndarray:
    """
    Calls _final_displacement_error() to get the oracle average displacement error.

    Args:
        ground_truth (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: oracle final displacement error (FDE), a single float number
    """

    return _final_displacement_error(ground_truth, pred, confidences, avails, "oracle")


def final_displacement_error_mean(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray,
) -> np.ndarray:
    """
    Calls _final_displacement_error() to get the mean average displacement error.

    Args:
        ground_truth (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: mean final displacement error (FDE), a single float number
    """

    return _final_displacement_error(ground_truth, pred, confidences, avails, "mean")

# TODO(perone): these functions were moved from the closed-loop evaluator.
# The functions _ego_agent_within_range, _get_boundingbox, _get_sides would ideally
# be moved to an abstraction of the Agent later. Such as in the example below:
#
#  ego = Agent(...)
#  agent = Agent(...)
#  bbox = ego.get_boundingbox()
#  within_range = ego.within_range(agent)
#  sides = ego.get_sides()


def _ego_agent_within_range(ego_centroid: np.ndarray, ego_extent: np.ndarray,
                            agent_centroid: np.ndarray, agent_extent: np.ndarray) -> np.ndarray:
    """This function will check if the agent is within range of the ego. It accepts
    as input a vectorized form with shapes N,D or a flat vector as well with shapes just D.

    :param ego_centroid: the ego centroid (shape: N, 2)
    :param ego_extent: the ego extent (shape: N, 3)
    :param agent_centroid: the agent centroid (shape: N, 2)
    :param agent_extent: the agent extent (shape: N, 3)
    :return: array with True if within range, False otherwise (shape: N)
    """
    distance = np.linalg.norm(ego_centroid - agent_centroid, axis=-1)
    norm_ego = np.linalg.norm(ego_extent[..., :2], axis=-1)
    norm_agent = np.linalg.norm(agent_extent[..., :2], axis=-1)
    max_range = 0.5 * (norm_ego + norm_agent)
    return distance < max_range


def _get_bounding_box(centroid: np.ndarray, yaw: np.ndarray,
                      extent: np.ndarray,) -> Polygon:
    """This function will get a shapely Polygon representing the bounding box
    with an optional buffer around it.

    :param centroid: centroid of the agent
    :param yaw: the yaw of the agent
    :param extent: the extent of the agent
    :return: a shapely Polygon
    """
    x, y = centroid[0], centroid[1]
    sin, cos = np.sin(yaw), np.cos(yaw)
    width, length = extent[0] / 2, extent[1] / 2

    x1, y1 = (x + width * cos - length * sin, y + width * sin + length * cos)
    x2, y2 = (x + width * cos + length * sin, y + width * sin - length * cos)
    x3, y3 = (x - width * cos + length * sin, y - width * sin - length * cos)
    x4, y4 = (x - width * cos - length * sin, y - width * sin + length * cos)
    return Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


# TODO(perone): this should probably return a namedtuple as otherwise it
# would have to depend on the correct ordering of the front/rear/left/right
def _get_sides(bbox: Polygon) -> Tuple[LineString, LineString, LineString, LineString]:
    """This function will get the sides of a bounding box.

    :param bbox: the bounding box
    :return: a tuple with the four sides of the bounding box as LineString,
             representing front/rear/left/right.
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.exterior.coords[:-1]
    return (
        LineString([(x1, y1), (x2, y2)]),
        LineString([(x3, y3), (x4, y4)]),
        LineString([(x1, y1), (x4, y4)]),
        LineString([(x2, y2), (x3, y3)]),
    )


class CollisionType(IntEnum):
    """This enum defines the three types of collisions: front, rear and side."""
    FRONT = 0
    REAR = 1
    SIDE = 2


def detect_collision(pred_centroid: np.ndarray, pred_yaw: np.ndarray,
                     pred_extent: np.ndarray, target_agents: np.ndarray) -> Optional[Tuple[CollisionType, str]]:
    """
    Computes whether a collision occured between ego and any another agent.
    Also computes the type of collision: rear, front, or side.
    For this, we compute the intersection of ego's four sides with a target
    agent and measure the length of this intersection. A collision
    is classified into a class, if the corresponding length is maximal,
    i.e. a front collision exhibits the longest intersection with
    egos front edge.

    .. note:: please note that this funciton will stop upon finding the first
              colision, so it won't return all collisions but only the first
              one found.

    :param pred_centroid: predicted centroid
    :param pred_yaw: predicted yaw
    :param pred_extent: predicted extent
    :param target_agents: target agents
    :return: None if not collision was found, and a tuple with the
             collision type and the agent track_id
    """
    ego_bbox = _get_bounding_box(centroid=pred_centroid, yaw=pred_yaw, extent=pred_extent)

    for agent in target_agents:
        # skip agents which are far from ego
        if not _ego_agent_within_range(pred_centroid, pred_extent,
                                       agent["centroid"], agent["extent"]):
            continue

        agent_bbox = _get_bounding_box(agent["centroid"], agent["yaw"], agent["extent"])

        if ego_bbox.intersects(agent_bbox):
            front_side, rear_side, left_side, right_side = _get_sides(ego_bbox)

            intersection_length_per_side = np.asarray(
                [
                    agent_bbox.intersection(front_side).length,
                    agent_bbox.intersection(rear_side).length,
                    agent_bbox.intersection(left_side).length,
                    agent_bbox.intersection(right_side).length,
                ]
            )
            argmax_side = np.argmax(intersection_length_per_side)

            # Remap here is needed because there are two sides that are
            # mapped to the same collision type CollisionType.SIDE
            max_collision_types = max(CollisionType).value
            remap_argmax = min(argmax_side, max_collision_types)
            collision_type = CollisionType(remap_argmax)
            return collision_type, agent["track_id"]
    return None
