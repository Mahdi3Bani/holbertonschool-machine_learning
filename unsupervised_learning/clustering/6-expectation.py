#!/usr/bin/env python3
'''Calculate the expectation step in the EM algorithm for a GMM'''
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in the EM algorithm for a GMM.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if m.shape[1] != X.shape[1]:
        return None, None
    if S.shape[1] != X.shape[1] or X.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None
    g = np.zeros((m.shape[0], X.shape[0]))
    for i in range(m.shape[0]):
        g[i] = pdf(X, m[i], S[i]) * pi[i]
    g_sum = np.sum(g, axis=0)
    g /= g_sum
    loo = np.sum(np.log(g_sum))
    return g, loo
