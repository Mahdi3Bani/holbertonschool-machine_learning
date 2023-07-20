#!/usr/bin/env python3
"""comment"""


import numpy as np


def markov_chain(P, s, t=1):
    '''comment'''
    n = P.shape[0]

    if P.shape != (n, n) or s.shape != (1, n)\
            or not np.allclose(np.sum(s), 1) or t < 0:
        return None

    return np.dot(s, np.linalg.matrix_power(P, t))
