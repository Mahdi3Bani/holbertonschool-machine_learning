#!/usr/bin/env python3
"""deep neural network performing binary classification"""


import numpy as np


class DeepNeuralNetwork:
    """deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for al in range(len(layers)):
            if type(layers[al]) != int or layers[al] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if al == 0:
                He = np.random.randn(layers[al], nx) * np.sqrt(2 / nx)
                self.weights['W' + str(al + 1)] = He
            else:
                He = np.random.randn(
                    layers[al], layers[al - 1]) * np.sqrt(2 / layers[al - 1])
                self.weights['W' + str(al + 1)] = He

            self.weights['b' + str(al + 1)] = np.zeros((layers[al], 1))
