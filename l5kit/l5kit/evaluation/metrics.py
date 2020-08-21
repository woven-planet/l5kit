import numpy as np


def _assert_shapes(gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray) -> None:
    """
    Check the shapes of args required by metrics

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:

    """
    assert len(pred.shape) == 3, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (num_modes,), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert np.allclose(np.sum(confidences), 1), "confidences should sum to 1"
    assert avails.shape == (future_len,), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert np.isfinite(pred).all(), "invalid value found in pred"
    assert np.isfinite(gt).all(), "invalid value found in gt"
    assert np.isfinite(confidences).all(), "invalid value found in confidences"
    assert np.isfinite(avails).all(), "invalid value found in avails"


def neg_multi_log_likelihood(
    gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray
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
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: negative log-likelihood for this example, a single float number
    """
    _assert_shapes(gt, pred, confidences, avails)

    gt = np.expand_dims(gt, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((gt - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    max_value = error.max()  # error are negative at this point, so max() gives the minimum one
    error = -np.log(np.sum(np.exp(error - max_value), axis=-1)) - max_value  # reduce modes
    return error


def rmse(gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray) -> np.ndarray:
    """
    Return the root mean squared error, computed using the stable nll

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: negative log-likelihood for this example, a single float number

    """
    nll = neg_multi_log_likelihood(gt, pred, confidences, avails)
    _, future_len, _ = pred.shape

    return np.sqrt(2 * nll / future_len)


def prob_true_mode(gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray) -> np.ndarray:
    """
    Return the probability of the true mode

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: a (modes) numpy array

    """
    _assert_shapes(gt, pred, confidences, avails)

    gt = np.expand_dims(gt, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((gt - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    max_value = error.max()  # error are negative at this point, so max() gives the minimum one

    error = np.exp(error - max_value) / np.sum(np.exp(error - max_value))
    return error


def time_displace(gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray) -> np.ndarray:
    """
    Return the displacement at time T

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: a (time) numpy array

    """
    true_mode_error = prob_true_mode(gt, pred, confidences, avails)
    true_mode_error = true_mode_error[:, None]  # add time axis

    gt = np.expand_dims(gt, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((gt - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    return np.sum(true_mode_error * np.sqrt(error), axis=0)  # reduce modes


def _average_displacement_error(
    gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray, mode: str
) -> np.ndarray:
    """
    Returns the average displacement error (ADE), which is the average displacement over all timesteps.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)

    Returns:
        np.ndarray: average displacement error (ADE), a single float number
    """
    _assert_shapes(gt, pred, confidences, avails)

    gt = np.expand_dims(gt, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((gt - pred) * avails) ** 2, axis=-1) ** 0.5  # reduce coords and use availability
    error = np.mean(error, axis=-1)  # average over timesteps

    if mode == "oracle":
        error = np.min(error)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=0)  # average over hypotheses
    else:
        print("Defaulting to mean calculation in _average_displacement_error().")
        error = np.mean(error, axis=0)  # average over hypotheses

    return error


def average_displacement_error_oracle(
    gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray,
) -> np.ndarray:
    """
    Calls _average_displacement_error() to get the oracle average displacement error.

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: oracle average displacement error (ADE), a single float number
    """

    return _average_displacement_error(gt, pred, confidences, avails, "oracle")


def average_displacement_error_mean(
    gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray,
) -> np.ndarray:
    """
    Calls _average_displacement_error() to get the mean average displacement error.

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: mean average displacement error (ADE), a single float number
    """

    return _average_displacement_error(gt, pred, confidences, avails, "mean")


def _final_displacement_error(
    gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray, mode: str
) -> np.ndarray:
    """
    Returns the final displacement error (FDE), which is the displacement calculated at the last timestep.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)

    Returns:
        np.ndarray: final displacement error (FDE), a single float number
    """
    _assert_shapes(gt, pred, confidences, avails)

    gt = np.expand_dims(gt, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((gt - pred) * avails) ** 2, axis=-1) ** 0.5  # reduce coords and use availability
    error = error[:, -1]  # use last timestep

    if mode == "oracle":
        error = np.min(error)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=0)  # average over hypotheses
    else:
        print("Defaulting to mean calculation in _final_displacement_error().")
        error = np.mean(error, axis=0)  # average over hypotheses

    return error


def final_displacement_error_oracle(
    gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray,
) -> np.ndarray:
    """
    Calls _final_displacement_error() to get the oracle average displacement error.

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: oracle final displacement error (FDE), a single float number
    """

    return _final_displacement_error(gt, pred, confidences, avails, "oracle")


def final_displacement_error_mean(
    gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray,
) -> np.ndarray:
    """
    Calls _final_displacement_error() to get the mean average displacement error.

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep

    Returns:
        np.ndarray: mean final displacement error (FDE), a single float number
    """

    return _final_displacement_error(gt, pred, confidences, avails, "mean")
