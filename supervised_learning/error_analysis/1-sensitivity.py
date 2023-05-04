#!/usr/bin/env python3
"""calculates the sensitivity for each class in a confusion matrix"""


import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix"""
    true_pos_false_neg = np.sum(confusion, axis=1)
    true_pos = np.diag(confusion)

    sens = true_pos / true_pos_false_neg
    return sens