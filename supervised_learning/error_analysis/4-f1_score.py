#!/usr/bin/env python3
"""calculates the F1 score for each class in a confusion matrix"""


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score for each class in a confusion matrix"""
    sens = sensitivity(confusion)
    prec = precision(confusion)
    return 2 * (prec * sens) / (prec + sens)
