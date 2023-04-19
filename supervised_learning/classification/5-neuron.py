#!/usr/bin/env python3
"""neuron class"""


import numpy as np


class Neuron:
    """"class that define a neuron for binary classification"""

    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        ''''The weights vector for the neuron'''
        return self.__W

    @property
    def b(self):
        ''''The bias for the neuron'''
        return self.__b

    @property
    def A(self):
        '''The activated output of the neuron'''
        return self.__A

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron'''
        z = np.dot(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-z))
        ''''sigmoid activation function'''
        return self.A

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression'''
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1.0000001-A))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron s predictions'''
        self.forward_prop(X)
        prediction = np.where(self.A >= 0.5, 1, 0)
        cost = self.cost(Y, self.A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''Calculates one pass of gradient descent on the neuron'''
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
