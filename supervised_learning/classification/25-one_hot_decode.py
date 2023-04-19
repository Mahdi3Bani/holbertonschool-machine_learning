#!/usr/bin/env python3
'''converts a numeric label vector into a one-hot matrix'''


import numpy as np


def one_hot_decode(one_hot):
    '''converts a one-hot matrix into a vector of labels'''
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    Y = np.argmax(one_hot, axis=0)
    return Y
