#!/usr/bin/env python3
"""class LSTMCell"""


import numpy as np


class LSTMCell:
    '''LSTMCell CLASS'''

    def __init__(self, i, h, o):
        '''CLASS constructor
            -i is the dimensionality of the data
            -h is the dimensionality of the hidden state
            -o is the dimensionality of the outputs'''
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(z):
        '''sigmoid funxtion'''
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        '''softmax'''
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        '''forward prog'''
        concat_x = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(np.dot(concat_x, self.Wf) + self.bf)
        u = self.sigmoid(np.dot(concat_x, self.Wu) + self.bu)

        calc_c = np.tanh(np.dot(concat_x, self.Wc) + self.bc)

        c_next = f * c_prev + u * calc_c

        o = self.sigmoid(np.dot(concat_x, self.Wo) + self.bo)

        h_next = o * np.tanh(c_next)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, c_next, y
