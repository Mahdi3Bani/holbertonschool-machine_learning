#!/usr/bin/env python3
'''converts a numeric label vector into a one-hot matrix'''


import numpy as np


def one_hot_encode(Y, classes):
    '''converts a numeric label vector into a one-hot matrix'''
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < 2 or classes < Y.max():
        return None
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot
