#!/usr/bin/env python3
"""definite"""


import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix:

        matrix is a numpy.ndarray of shape (n, n) whose
        definiteness should be calculated

        If matrix is not a numpy.ndarray, raise a TypeError
        with the message matrix must be a numpy.ndarray

        If matrix is not a valid matrix, return None

        Return: the string Positive definite, Positive
        semi-definite, Negative semi-definite, Negative definite,
        or Indefinite if the matrix is positive definite, positive
        semi-definite, negative semi-definite, negative definite of
        indefinite, respectively

        If matrix does not fit any of the above categories, return None
"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues = np.linalg.eigvals(matrix)
    positive_eigenvalues = np.sum(eigenvalues > 0)
    zero_eigenvalues = np.sum(eigenvalues == 0)

    if positive_eigenvalues == matrix.shape[0]:
        return "Positive definite"
    elif positive_eigenvalues > 0 and zero_eigenvalues > 0:
        return "Positive semi-definite"
    elif positive_eigenvalues == 0 and zero_eigenvalues > 0:
        return "Negative semi-definite"
    elif positive_eigenvalues == 0 and zero_eigenvalues == 0:
        return "Negative definite"
    else:
        return "Indefinite"
