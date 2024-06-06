#!/usr/bin/env python3
"""RNN Encoder class"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """init func"""
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
        """Initialize hidden layers to 0
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Forward pass through the layers.
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
