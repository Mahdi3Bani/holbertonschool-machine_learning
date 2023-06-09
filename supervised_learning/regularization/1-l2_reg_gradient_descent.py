#!/usr/bin/env python3
"""
    Updates the weights and biases of a neural
    network using gradient descent with L2 regularization.
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural
    network using gradient descent with L2 regularization.
    """

    m = Y.shape[1]

    dZ = cache['A'+str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A'+str(i-1)]
        W = weights['W'+str(i)]
        b = weights['b'+str(i)]
        dW = (1/m) * np.dot(dZ, A.T) + (lambtha/m) * W
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.dot(W.T, dZ) * (1 - np.power(A, 2)
                                ) if i > 1 else np.dot(W.T, dZ)
        weights['W'+str(i)] -= alpha * dW
        weights['b'+str(i)] -= alpha * db
