#!/usr/bin/env python3
"""
    Calculates the maximization step in the EM algorithm for a GMM
"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if g.shape[1] != X.shape[0]:
        return None, None, None
    cluster = np.sum(g, axis=0)
    cluster = np.sum(cluster)
    if int(cluster) != X.shape[0]:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        pi[i] = np.sum(g[i])/n
        m[i] = np.matmul(g[i], X)/np.sum(g[i])
        distance = X - m[i]
        S[i] = np.matmul(g[i] * distance.T, distance)/np.sum(g[i])
    return pi, m, S
