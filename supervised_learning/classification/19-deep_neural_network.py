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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for al in range(len(layers)):
            if type(layers[al]) != int or layers[al] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if al == 0:
                He = np.random.randn(layers[al], nx) * np.sqrt(2 / nx)
                self.__weights['W' + str(al + 1)] = He
            else:
                He = np.random.randn(
                    layers[al], layers[al - 1]) * np.sqrt(2 / layers[al - 1])
                self.__weights['W' + str(al + 1)] = He

            self.__weights['b' + str(al + 1)] = np.zeros((layers[al], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron'''
        self.__cache["A0"] = X
        for i in range(self.__L):
            z = np.dot(self.__weights['W' + str(i + 1)],
                       self.__cache['A'+str(i)]) +\
                self.__weights['b'+str(i + 1)]
            A = 1 / (1 + np.exp(-z))
            ''''sigmoid activation function'''
            self.__cache['A' + str(i + 1)] = A
        return A, self.__cache

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression'''
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1.0000001-A))
        return cost
