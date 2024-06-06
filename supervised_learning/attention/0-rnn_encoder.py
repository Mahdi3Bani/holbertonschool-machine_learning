#!/usr/bin/env python3
"""rnn encoder class"""


import tensorflow as tf

class RNNEncoder(tf.keras.layers.Layer):
    '''enn encoder class'''
    def __init__(self, vocab, embedding, units, batch):
        '''init function'''
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')


    def initialize_hidden_state(self):
        '''initialize hidden layers to 0'''
        return tf.zeros((self.batch, self.units))


    def call(self, x, initial):
        """forward pass through the layers"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
