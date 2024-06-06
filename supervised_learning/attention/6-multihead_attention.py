#!/usr/bin/env python3
"""Multi-head Attention class."""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention class."""

    def __init__(self, dm, h):
        """init func
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        if dm % h != 0:
            dm = dm // h * h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def reshape(self, x, batch):
        """Reshape inputs.
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(self, Q, K, V, mask):
        """Forward pass through the layers
        """
        batch = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = self.reshape(q, batch)
        k = self.reshape(k, batch)
        v = self.reshape(v, batch)
        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        output = self.linear(output)
        return output, weights
