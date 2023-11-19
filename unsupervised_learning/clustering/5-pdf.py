#!/usr/bin/env python3
""" calculate the probability density function of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """
    Calculate the probability density function of a Gaussian distribution.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1\
            or m.shape[0] != X.shape[1]:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2 \
            or S.shape[0] != S.shape[1] or S.shape[0] != X.shape[1]:
        return None

    dimension = len(m)
    covariance_det = np.linalg.det(S)

    if covariance_det < 1:
        return None

    covariance_inv = np.linalg.inv(S)
    diff = X - m

    exponent = -0.5 * np.sum(np.dot(diff, covariance_inv) * diff, axis=1)
    prefactor = 1 / ((2 * np.pi) ** (dimension / 2) * np.sqrt(covariance_det))

    P = prefactor * np.exp(exponent)

    P = np.maximum(P, 1e-300)

    return P
