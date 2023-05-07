#!/usr/bin/env python3

import numpy as np



def softmax(z):
    """
    Applies the softmax activation function elementwise to the input array z.

    Args:
        z (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying the softmax function.
    """
    
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    """
    cache = {'A0': X}
    
    for l in range(L):
        A = cache['A' + str(l)]
        W = weights['W' + str(l + 1)]
        b = weights['b' + str(l + 1)]
        Z = np.dot(W, A) + b
        if l != L - 1:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=Z.shape)
            A *= D
            A /= keep_prob
            cache['D' + str(l + 1)] = D
        else:
            A = softmax(Z)
        cache['A' + str(l + 1)] = A
    return cache
