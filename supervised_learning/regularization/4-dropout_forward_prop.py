#!/usr/bin/env python3
"""Conducts forward propagation using Dropout."""


import numpy as np


def softmax(z):
    """
    Applies the softmax activation function elementwise to the input array z.
    """

    return np.exp(z) / np.sum(np.exp(z), axis=0)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    """
    cache = {'A0': X}

    for layer in range(L):
        A = cache['A' + str(layer)]
        W = weights['W' + str(layer + 1)]
        b = weights['b' + str(layer + 1)]
        Z = np.dot(W, A) + b
        if layer != L - 1:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=Z.shape)
            A *= D
            A /= keep_prob
            cache['D' + str(layer + 1)] = D
        else:
            A = softmax(Z)
        cache['A' + str(layer + 1)] = A
    return cache
