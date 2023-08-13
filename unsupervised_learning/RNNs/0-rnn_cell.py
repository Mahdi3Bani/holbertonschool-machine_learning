#!/usr/bin/env python3
"""class RNN"""


import numpy as np


class RNNCell:
    '''RNN CLASS'''

    def __init__(self, i, h, o):
        '''CLASS constructor
            -i is the dimensionality of the data
            -h is the dimensionality of the hidden state
            -o is the dimensionality of the outputs'''
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(z):
        '''softmax funxtion'''
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat_input, self.Wh) + self.bh)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y
