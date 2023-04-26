#!/usr/bin/env python3
"""shuffles the data points in two matrices the same way"""


import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""

    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled
