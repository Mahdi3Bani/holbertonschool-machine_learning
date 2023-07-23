#!/usr/bin/env python3


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """comment"""
    T = Observation.shape[0]
    N, M = Emission.shape

    if T == 0 or N == 0 or M == 0:
        return None, None

    viterbi_matrix = np.zeros((N, T))
    backpointer_matrix = np.zeros((N, T), dtype=int)

    viterbi_matrix[:, 0] = Initial.reshape(-1) * Emission[:, Observation[0]]

    for t in range(1, T):
        for i in range(N):
            probabilities = viterbi_matrix[:, t - 1] * \
                Transition[:, i] * Emission[i, Observation[t]]
            backpointer_matrix[i, t] = np.argmax(probabilities)
            viterbi_matrix[i, t] = np.max(probabilities)

    best_path_prob = np.max(viterbi_matrix[:, -1])
    best_last_state = np.argmax(viterbi_matrix[:, -1])

    best_path = [best_last_state]
    for t in range(T - 1, 0, -1):
        best_last_state = backpointer_matrix[best_last_state, t]
        best_path.insert(0, best_last_state)

    return best_path, best_path_prob
