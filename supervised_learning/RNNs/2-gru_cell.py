#!/usr/bin/env python3
"""class GRUCell"""


import numpy as np


class GRUCell:
    '''GRUCell CLASS'''

    def __init__(self, i, h, o):
        '''CLASS constructor
            -i is the dimensionality of the data
            -h is the dimensionality of the hidden state
            -o is the dimensionality of the outputs'''
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
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
        z = self.sigmoid(np.dot(concat_x, self.Wz) + self.bz)
        r = self.sigmoid(np.dot(concat_x, self.Wr) + self.br)

        concat_prev_h = np.concatenate((r * h_prev, x_t), axis=1)
        calc_h = np.tanh(np.dot(concat_prev_h, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * calc_h
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y
