#!/usr/bin/env python3
"""class BidirectionalCell"""


import numpy as np


class BidirectionalCell:
    '''BidirectionalCell CLASS'''

    def __init__(self, i, h, o):
        '''CLASS constructor
            -i is the dimensionality of the data
            -h is the dimensionality of the hidden state
            -o is the dimensionality of the outputs'''
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h * 2, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(z):
        '''sigmoid funxtion'''
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        '''softmax'''
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        '''forward prog'''
        concat_x = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(concat_x, self.Whf) + self.bhf)
        return h_next
    
    def backward(self, h_next, x_t):
        x = np.concatenate((h_next, x_t), axis=1)

        h_prev = np.tanh(np.dot(x, self.Whb) + self.bhb)

        return h_prev
