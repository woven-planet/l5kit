import math
from typing import Tuple

import numpy as np
from scipy import optimize

from l5kit.geometry import angular_distance


def fit_ackerman_model_approximate(
    gx: np.ndarray,
    gy: np.ndarray,
    gr: np.ndarray,
    gv: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    wr: np.ndarray,
    wv: np.ndarray,
    wgx: np.ndarray,
    wgy: np.ndarray,
    wgr: np.ndarray,
    wgv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits feasible ackerman-steering trajectory to groundtruth control points.
    Groundtruth is represented as 4 input numpy arrays ``(gx, gy, gr, gv)``
    each of size ``(N,)`` representing position, rotation and velocity at time ``i``.
    Returns 4 arrays ``(x, y, r, v)`` each of shape ``(N,)`` - the optimal trajectory.
    The solution is found as minimization of the following non-linear least squares problem:
    minimize F(x, y, r, v) = F_ground_cost(x, y, r, v) + F_kinematics_cost(x, y, r, v) where
    F_ground_cost(x, y, r, v) = 0.5 * sum(
    (wgx[i] * (x[i] - gx[i])) ** 2 +
    (wgy[i] * (y[i] - gy[i])) ** 2 +
    (wgr[i] * (r[i] - gr[i])) ** 2 +
    (wgv[i] * (v[i] - gv[i])) ** 2,
    i = 0 ... N-1)
    and
    F_kinematics_cost(x, y, r, v) = 0.5 * sum(
    (wx * (x[i] + cos(r[i]) * v[i] - x[i+1])) ** 2 +
    (wy * (y[i] + sin(r[i]) * v[i] - y[i+1])) ** 2 +
    (wr * (r[i] - r[i+1])) ** 2 +
    (wv * (v[i] - v[i+1])) ** 2,
    i = 0 ... N-2)
    Weights wg* control adherance to the control points while
    weights w* control obeying of underlying kinematic motion constrains.

    :return: 4 arrays (x, y, r, v) each of shape (N,), the optimal trajectory.
    """

    N = len(gx)

    w = np.hstack([wgx, wgy, wgr, wgv, wx, wy, wr, wv])

    def residuals(xyrv: np.ndarray) -> np.ndarray:
        x, y, r, v = np.split(xyrv, 4)

        x1, x2 = x[0:N - 1], x[1:N]
        y1, y2 = y[0:N - 1], y[1:N]
        r1, r2 = r[0:N - 1], r[1:N]
        v1, v2 = v[0:N - 1], v[1:N]

        return w * np.hstack(
            [
                x - gx,
                y - gy,
                angular_distance(r, gr),
                v - gv,
                np.append(x1 + np.cos(r1) * v1 - x2, 0),
                np.append(y1 + np.sin(r1) * v1 - y2, 0),
                np.append(angular_distance(r1, r2), 0),
                np.append(v1 - v2, 0),
            ]
        )

    # jacobian of residuals
    def jacobian(xyrv: np.ndarray) -> np.ndarray:
        x, y, r, v = np.split(xyrv, 4)

        z = np.zeros((N, N))
        e = np.eye(N, N)
        e0 = np.block([[np.eye(N - 1, N - 1), np.zeros((N - 1, 1))], [np.zeros((1, N))]])
        e1 = np.block([[np.zeros((N - 1, 1)), np.eye(N - 1, N - 1)], [np.zeros((1, N))]])

        A = np.block(
            [
                [e, z, z, z],
                [z, e, z, z],
                [z, z, e, z],
                [z, z, z, e],
                [e0 - e1, z, -np.sin(r) * v * e0, np.cos(r) * e0],
                [z, e0 - e1, np.cos(r) * v * e0, np.sin(r) * e0],
                [z, z, e0 - e1, z],
                [z, z, z, e0 - e1],
            ]
        )

        return w[:, None] * A

    # Gauss–Newton algorithm
    xyrv = np.hstack([gx, gy, gr, gv])
    for _ in range(5):
        xyrv = xyrv - np.linalg.lstsq(jacobian(xyrv), residuals(xyrv), rcond=None)[0]
    x, y, r, v = np.split(xyrv, 4)
    return x, y, r, v


def fit_ackerman_model_exact(
    x0: np.ndarray,
    y0: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gr: np.ndarray,
    gv: np.ndarray,
    wgx: np.ndarray,
    wgy: np.ndarray,
    wgr: np.ndarray,
    wgv: np.ndarray,
    ws: float = 5.0,
    wa: float = 5.0,
    min_acc: float = -0.3,  # min acceleration: -3 mps2
    max_acc: float = 0.3,   # max acceleration: 3 mps2
    min_steer: float = -math.radians(45) * 0.1,  # max yaw rate: 45 degrees per second
    max_steer: float = math.radians(45) * 0.1,   # max yaw rate: 45 degrees per second
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits feasible ackerman-steering trajectory to groundtruth control points.
    Groundtruth is represented as 4 numpy arrays ``(gx, gy, gr, gv)``
    each of shape ``(N,)`` representing position, rotation and velocity at time i.
    Returns 4 arrays ``(x, y, r, v)`` each of shape ``(N,)`` - the optimal trajectory.

    The solution is found as minimisation of the following non-linear least squares problem:
    ::
    minimize F(steer, acc) = 0.5 * sum(
    (wgx[i] * (x[i] - gx[i])) ** 2 +
    (wgy[i] * (y[i] - gy[i])) ** 2 +
    (wgr[i] * (r[i] - gr[i])) ** 2 +
    (wgv[i] * (v[i] - gv[i])) ** 2 +
    (ws * steer[i]) ** 2 +
    (wa * acc[i]) ** 2)
    i = 1 ... N)
    subject to following unicycle motion model equations:
    x[i+1] = x[i] + cos(r[i]) * v[i]
    y[i+1] = y[i] + sin(r[i]) * v[i]
    r[i+1] = r[i] + steer[i]
    v[i+1] = v[i] + acc[i]
    min_steer < steer[i] < max_steer
    min_acc < acc[i] < max_acc
    for i = 0 .. N
    Weights ``wg*`` control adherence to the control points
    In a typical usecase ``wgx = wgy = 1`` and ``wgr = wgv = 0``

    :return: 4 arrays ``(x, y, r, v)`` each of shape ``(N,)``- the optimal trajectory.
    """
    N = len(gx)

    wsteer_acc = np.hstack([ws * np.ones(N), wa * np.ones(N)])

    def control2position(steer_acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        steer, acc = np.split(steer_acc, 2)
        r = r0 + np.cumsum(steer)
        v = v0 + np.cumsum(acc)
        x = x0 + np.cumsum(np.cos(r) * v)
        y = y0 + np.cumsum(np.sin(r) * v)
        return x, y, r, v

    def residuals(steer_acc: np.ndarray) -> np.ndarray:
        x, y, r, v = control2position(steer_acc)
        return np.hstack(
            [
                wgx * (x - gx),
                wgy * (y - gy),
                wgr * angular_distance(r, gr),
                wgv * (v - gv),
                wsteer_acc * steer_acc,
            ]
        )

    def jacobian(steer_acc: np.ndarray) -> np.ndarray:
        x, y, r, v = control2position(steer_acc)

        Jr1, Jr2, Jr3, Jr4 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        Jv1, Jv2, Jv3, Jv4 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        J = np.zeros((2 * N, 4 * N))

        for i in range(N - 1, -1, -1):
            Jr1[i:] = Jr1[i:] - np.sin(r[i]) * v[i] * wgx[i:]
            Jr2[i:] = Jr2[i:] + np.cos(r[i]) * v[i] * wgy[i:]
            Jv1[i:] = Jv1[i:] + np.cos(r[i]) * wgx[i:]
            Jv2[i:] = Jv2[i:] + np.sin(r[i]) * wgy[i:]
            Jr3[i:] = wgr[i:]
            Jv4[i:] = wgv[i:]

            J[i, :] = np.hstack([Jr1, Jr2, Jr3, Jr4])
            J[N + i, :] = np.hstack([Jv1, Jv2, Jv3, Jv4])

        return np.vstack([J.T, np.eye(N + N) * wsteer_acc])

    min_bound = np.concatenate((min_steer * np.ones(N), min_acc * np.ones(N)))
    max_bound = np.concatenate((max_steer * np.ones(N), max_acc * np.ones(N)))
    result = optimize.least_squares(residuals, np.zeros(2 * N), jacobian, (min_bound, max_bound))

    x, y, r, v = control2position(result["x"])
    steer, acc = result["x"][:N], result["x"][N:]
    return x, y, r, v, acc, steer
