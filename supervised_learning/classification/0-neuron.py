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

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
