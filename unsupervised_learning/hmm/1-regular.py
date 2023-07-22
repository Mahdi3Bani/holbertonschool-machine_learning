#!/usr/bin/env python3
"""comment"""


import numpy as np


def regular(P):
    """comment"""
    n = P.shape[0]

    if n != P.shape[1]:
        return None

    if not np.all(np.linalg.matrix_power(P, n) > 0):
        return None

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    index = np.where(np.isclose(eigenvalues, 1.0))[0]

    if len(index) != 1:
        return None

    steady_state_probabilities = np.real(
        eigenvectors[:, index] / np.sum(eigenvectors[:, index]))
    return steady_state_probabilities.flatten()
