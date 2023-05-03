#!/usr/bin/env python3
"""creates a confusion matrix"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix"""
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    for i in range(m):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        confusion[true_label, predicted_label] += 1
    return confusion
