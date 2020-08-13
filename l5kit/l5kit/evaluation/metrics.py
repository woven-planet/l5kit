import numpy as np


def neg_multi_log_likelihood(
    gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray
) -> np.ndarray:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/

    Args:
        gt (np.ndarray): array of shape (time)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (time) with the availability for each gt timestep
    Returns:
        np.ndarray: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 3, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (future_len, num_coords), f"expected 2D (TxC) array for gt, got {gt.shape}"
    assert confidences.shape == (num_modes,)
    assert np.allclose(np.sum(confidences), 1), "confidences should sum to 1"
    assert avails.shape == (future_len,)

    gt = np.expand_dims(gt, 0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((gt - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce time
    # use max aggregator on modes for numerical stability
    max_value = error.max()  # error are negative at this point, so max() gives the minimum one
    error = -np.log(np.sum(np.exp(error - max_value), axis=-1)) - max_value  # reduce modes
    return error
