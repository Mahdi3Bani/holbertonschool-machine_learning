#!/usr/bin/env python3
"""MultiNormal"""


import numpy as np
mean_cov = __import__('0-mean_cov').mean_cov


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

        mean, cov = mean_cov(data.T)
        self.mean = mean.T
        self.cov = cov

    def pdf(self, x):
        '''
        calculates the PDF at a data point:
            *x is a numpy.ndarray of shape (d, 1) containing the
            data point whose PDF should be calculated
                -d is the number of dimensions of the Multinomial instance

            *If x is not a numpy.ndarray, raise a TypeError with
            the message x must be a numpy.ndarray

            *If x is not of shape (d, 1), raise a ValueError with the
            message x must have the shape ({d}, 1)

        Returns the value of the PDF
        '''
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if len(x.shape) != 2 or x.shape != (self.cov.shape[0], 1):
            raise ValueError(
                'x must have the shape ({}, 1)'.format(self.cov.shape[0]))

        pdf = (1 / np.sqrt(((2 * np.pi) ** self.cov.shape[0]) * np.linalg.det(self.cov)) *
               np.exp(-(np.linalg.solve(self.cov, x - self.mean).
                        T.dot(x - self.mean)) / 2))

        return pdf[0][0]
