#!/usr/bin/env python3
'''comment'''


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    '''comment'''
    T = Observation.shape[0]
    N, M = Emission.shape

    if T == 0 or N == 0 or M == 0:
        return None, None

    backward_matrix = np.zeros((N, T))

    backward_matrix[:, -1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            probabilities = Transition[i, :] * Emission[:,
                            Observation[t + 1]] * backward_matrix[:, t + 1]
            backward_matrix[i, t] = np.sum(probabilities)

    P = np.sum(Initial.reshape(-1) *
               Emission[:, Observation[0]] * backward_matrix[:, 0])

    return P, backward_matrix
