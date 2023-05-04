#!/usr/bin/env python3
"""calculates the specificity for each class in a confusion matrix"""


import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix"""
    true_neg_false_pos = np.sum(confusion, axis=0) - np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - np.diag(confusion)

    spec = true_neg_false_pos / (true_neg_false_pos + false_neg)
    return spec
