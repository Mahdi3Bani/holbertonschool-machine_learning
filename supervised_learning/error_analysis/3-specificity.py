#!/usr/bin/env python3
"""calculates the specificity for each class in a confusion matrix"""


import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix"""
    spec = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        true_neg = np.sum(confusion) - np.sum(confusion[i, :])\
            - np.sum(confusion[:, i]) + confusion[i, i]
        false_pos = np.sum(confusion[:, i]) - confusion[i, i]

        spec[i] = true_neg / (true_neg + false_pos)
    return spec
