#!/usr/bin/env python3



import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    """

    outputs = {}
    masks = {}
    outputs['A0'] = X

    for l in range(1, L+1):
        Z = np.dot(weights['W' + str(l)], outputs['A' + str(l-1)]) + weights['b' + str(l)]

        if l < L:
            A = np.tanh(Z)
            mask = np.random.binomial(1, keep_prob, size=A.shape)
            A *= mask
            A /= keep_prob
            masks['D' + str(l)] = mask
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

        outputs['Z' + str(l)] = Z
        outputs['A' + str(l)] = A

    return outputs, masks
