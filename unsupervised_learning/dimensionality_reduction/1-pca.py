#!/usr/bin/env python3
"""comment"""


import numpy as np


def pca(X, ndim):
    """comment"""
    X_n = X - np.mean(X)

    _, _, vh = np.linalg.svd(X_n)

    W = vh.T[:, :ndim]

    return np.dot(X_n, W)

