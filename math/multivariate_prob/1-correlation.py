#!/usr/bin/env python3
"""Correlation"""


import numpy as np


def correlation(C):
    """
    calculates a correlation matrix:
        *C is a numpy.ndarray of shape (d, d) containing a covariance matrix:
            -d is the number of dimensions
            -f C is not a numpy.ndarray, raise a TypeError with the message
            C must be a numpy.ndarray
            -If C does not have shape (d, d), raise a ValueError with the
            message C must be a 2D square matrix

        *Returns a numpy.ndarray of shape (d, d) containing the
        correlation matrix

    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    correlation_matrix = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            correlation_matrix[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])

    return correlation_matrix
