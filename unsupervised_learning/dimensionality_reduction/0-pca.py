#!/usr/bin/env python3
"""comment"""


import numpy as np


def pca(X, var=0.95):
    """comment"""
    _, s, vh = np.linalg.svd(X)

    var_c = np.cumsum(s) / np.sum(s)

    r = np.argwhere(var_c >= var)[0, 0]

    W = vh[:r + 1].T
    return W
