#!/usr/bin/env python3
'''comment'''


import numpy as np


def absorbing(P):
    """comment"""
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None

    if not (np.all(P >= 0) and np.all(P <= 1)):
        return None

    if not (np.all(np.sum(P, axis=1) == 1)):
        return None

    if P.shape[0] < 1:
        return None

    state_matrix = np.dot(P, P)

    for i in range(state_matrix.shape[0]):
        diagonal_element = state_matrix[i, i]

        if diagonal_element == 1:
            submatrix = state_matrix[i:]

            if not np.any(submatrix[:, i] == 0):
                return True

    return False
