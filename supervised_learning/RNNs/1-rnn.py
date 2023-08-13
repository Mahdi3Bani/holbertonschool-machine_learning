#!/usr/bin/env python3
"""function RNN"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    '''performs forward propagation for a simple RNN'''
    t, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    H[0] = h_0

    for i in range(t):
        h_prev = H[i]
        x_t = X[i]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[i + 1] = h_next
        Y[i] = y
    return H, Y
