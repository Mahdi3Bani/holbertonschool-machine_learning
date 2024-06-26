#!/usr/bin/env python3
"""performs the expectation maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol <= 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    g, lk = expectation(X, pi, m, S)
    prev_lk = 0
    text = "Log Likelihood after {} iterations: {}"
    for i in range(iterations):
        if verbose and (i % 10 == 0):
            print(text.format(i, lk.round(5)))
        pi, m, S = maximization(X, g)
        g, lk = expectation(X, pi, m, S)
        if abs(lk - prev_lk) <= tol:
            break
        prev_lk = lk
    if verbose:
        print(text.format(i + 1, lk.round(5)))
    return pi, m, S, g, lk
