#!/usr/bin/env python3
"""comment"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    '''comment'''
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    n, d = X.shape
    if n < k:
        return None, None, None

    m, clusters = kmeans(X, k)
    pi = np.ones(shape=(k,))/k
    s = np.tile(np.identity(X.shape[1])[np.newaxis, :, :], (k, 1, 1))

    return pi, m, s
