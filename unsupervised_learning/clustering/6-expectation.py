#!/usr/bin/env python3
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in the EM algorithm for a GMM.
    """

    n, d = X.shape
    k = m.shape[0]

    # Initialize the posterior probability matrix
    g = np.zeros((k, n))

    for i in range(k):

        g[i] = pdf(X, m[i], S[i]) * pi[i]

    g_sum = np.sum(g, axis=0)
    g /= g_sum

    l = np.sum(np.log(g_sum))

    return g, l
