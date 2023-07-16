#!/usr/bin/node
"""comment"""

import numpy as np


def kmeans(X, k, iterations=1000):
    '''comment'''
    if iterations < 1:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    n, d = X.shape
    if n < k:
        return None, None

    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    C = np.random.uniform(low=min_values, high=max_values, size=(k, d))

    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):
        prev_C = np.copy(C)

        dists = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dists, axis=1)

        for i in range(k):
            if np.sum(clss == i) > 0:
                C[i] = np.mean(X[clss == i], axis=0)
            else:
                C[i] = np.random.uniform(min_values, max_values)

        if np.allclose(C, prev_C):
            return C, clss

    return None, None
