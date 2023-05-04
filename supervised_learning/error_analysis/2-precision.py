#!/usr/bin/env python3
"""calculates the precision for each class in a confusion matrix"""


import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix"""
    true_pos_false_pos = np.sum(confusion, axis=0)
    true_pos = np.diag(confusion)

    prec = true_pos / true_pos_false_pos
    return prec
