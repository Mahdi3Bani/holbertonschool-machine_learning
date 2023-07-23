#!/usr/bin/env python3
"""comment"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None

    T = Observation.shape[0]

    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    N, _ = Emission.shape

    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return None, None

    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None

    F = np.zeros((N, T))
    for i in range(N):
        F[i, 0] = Initial[i] * Emission[i, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t - 1] * Transition[:, j] * Emission[j, Observation[t]])

    P = np.sum(F[:, T - 1])

    return P, F
