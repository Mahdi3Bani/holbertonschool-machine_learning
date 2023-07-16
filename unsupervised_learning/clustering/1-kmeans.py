#!/usr/bin/env python3
"""comment"""

import numpy as np


def initialize(X, k):
    '''comment'''
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    n, d = X.shape
    if n < k:
        return None

    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_values, high=max_values, size=(k, d))
    return centroids


def kmeans(X, k, iterations=1000):
    '''comment'''

    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    n, d = X.shape
    if n < k:
        return None, None

    C = initialize(X, k)

    for _ in range(iterations):
        prev_C = np.copy(C)

        dists = np.sqrt(np.sum((X - C[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(dists, axis=0)

        for i in range(k):
            if np.sum(clss == i) > 0:
                C[i] = np.mean(X[clss == i], axis=0)
            else:
                C[i] = initialize(X, 1)

        if np.allclose(C, prev_C):
            return C, clss

    return None, None
