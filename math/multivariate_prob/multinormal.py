#!/usr/bin/env python3
"""MultiNormal"""


import numpy as np


class MultiNormal:
    """
    class MultiNormal that represents a Multivariate Normal distribution:
        *class constructor def __init__(self, data):
            -data is a numpy.ndarray of shape (d, n) containing the
            data set:
            -n is the number of data points
            -d is the number of dimensions in each data point
            -If data is not a 2D numpy.ndarray, raise a TypeError
            with the message data must be a 2D numpy.ndarray
            -If n is less than 2, raise a ValueError with the message
            data must contain multiple data points
        *Set the public instance variables:
            -mean - a numpy.ndarray of shape (d, 1)
            containing the mean of data
            -cov - a numpy.ndarray of shape (d, d) containing
            the covariance matrix data

    """

    def __init__(self, data):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        _, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        centered = data - self.mean
        self.cov = np.dot(centered, centered.T) / (n - 1)
