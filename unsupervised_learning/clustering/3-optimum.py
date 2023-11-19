#!/usr/bin/env python3
"""testing the optimal number of clusters"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''function to find the optimal value'''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    
    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None


    if kmax is None:
        kmax = X.shape[0]

    if kmin < 1 or kmax < kmin or iterations < 1:
        return None, None

    prev_var = 0.0
    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        centroids, clusters = kmeans(X, k, iterations)
        v = variance(X, centroids)
        x = prev_var - v
        results.append((centroids, clusters))
        if x <= 0:
            prev_var = v
            x = 0.0

        d_vars.append(x)

    return results, np.array(d_vars).tolist()
