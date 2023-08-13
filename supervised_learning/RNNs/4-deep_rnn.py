#!/usr/bin/env python3
"""function deep_rnn"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    '''performs forward propagation for a  deep_rnn'''
    t, m, i = X.shape
    l = len(rnn_cells)
    _, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    Y = []

    H[0] = h_0

    for i in range(t):
        x_t = X[i]
        for l in range(l):
            if l == 0:
                h_prev = H[i, l]
            else:
                h_prev = H[i + 1, l - 1]
            h_next, _ = rnn_cells[l].forward(h_prev, x_t)
            H[i + 1, l] = h_next
        Y[i] = Y[i + 1, -1]
    return H, Y
