#!/usr/bin/env python3
"""RNN Decoder class."""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class."""

    def __init__(self, vocab, embedding, units, batch):
        """init func"""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.batch = batch
        self.units = units
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """Forward pass through the layers"""
        context, _ = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        X = tf.concat([tf.cast(context, dtype=tf.float32),
                       tf.cast(x, dtype=tf.float32)], axis=-1)
        y, hidden = self.gru(X)
        Y = tf.reshape(y, (-1, y.shape[2]))
        Y = self.F(Y)
        return Y, hidden
