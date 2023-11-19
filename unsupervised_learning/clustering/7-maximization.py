#!/usr/bin/env python3
"""
    Calculates the maximization step in the EM algorithm for a GMM
"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    """
    n, d = X.shape
    k = g.shape[0]

    pi = np.sum(g, axis=1) / n

    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    cov_mat = np.zeros((k, d, d))

    for j in range(k):
        diff = X - m[j]
        cov_mat[j] = np.dot(g[j] * diff.T, diff) / np.sum(g[j])

    return pi, m, S
