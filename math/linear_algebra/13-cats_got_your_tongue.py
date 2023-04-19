#!/usr/bin/env python3
"""concat 2 np arrays"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concat 2 np arrays"""

    return np.concatenate((mat1, mat2), axis=axis)
