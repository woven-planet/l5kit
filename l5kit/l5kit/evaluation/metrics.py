import numpy as np


def single_trajectory_metrics(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute the mean over sample of the euclidean distance between gt and pred coords.
    Time dimension is not reduced

    Args:
        gt (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(time)x(2D coords)

    Returns:
        np.ndarray: array of shape (time) with the average displacement at each time-step

    """
    assert gt.shape == pred.shape, "print gt and pred shape don't match"
    assert len(gt.shape) == 3, f"expected 3D (BxTxC) array for gt, got {len(gt.shape)}"
    assert len(pred.shape) == 3, f"expected 3D (BxTxC) array for pred, got {len(pred.shape)}"
    eucl_dis = np.linalg.norm(gt - pred, axis=-1)  # reduce coords
    error = eucl_dis.mean(axis=0)  # reduce samples, keep time
    return error


def multi_trajectory_metrics(gt: np.ndarray, pred: np.ndarray, confidences: np.ndarray) -> np.ndarray:
    """
    Compute a negative log-likelihood for the multi-mode scenario.
    # TODO insert formula here

    Args:
        gt (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
    Returns:
        np.ndarray: array of shape (1,)

    """

    assert len(gt.shape) == 3, f"expected 3D (BxTxC) array for gt, got {len(gt.shape)}"
    assert len(pred.shape) == 4, f"expected 3D (BxNxTxC) array for pred, got {len(pred.shape)}"
    assert pred.shape[:2] == confidences.shape, "expected a confidence value for every trajectory"
    assert np.allclose(np.sum(confidences, axis=-1), 1), "confidences should sum to 1 for each sample"

    gt = np.expand_dims(gt, 1)

    eucl_dis = np.linalg.norm(gt - pred, axis=-1)  # reduce coords

    error = confidences * np.exp(-0.5 * eucl_dis.sum(axis=-1))  # reduce time
    error = -np.log(np.sum(error, axis=-1))  # reduce modes
    error = np.mean(error)  # reduce samples
    return error


if __name__ == "__main__":
    pred = np.zeros((1, 2, 3, 1))
    pred[0, 0] = [[0], [0], [0]]
    pred[0, 1] = [[10], [10], [10]]

    gt = np.ones((1, 3, 1))
    gt[0] = [[5], [5], [5]]

    confs = np.random.rand(1, 2)
    confs[0] = [1, 0]

    err = multi_trajectory_metrics(gt, pred, confs)
    print(err)
