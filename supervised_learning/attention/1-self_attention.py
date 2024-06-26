#!/usr/bin/env python3
"""Create Attention class."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Self-Attention class."""

    def __init__(self, units):
        """init func
        """
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Self-attention mechanism.
        """
        s_prev_with_time_axis = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(
            self.W(s_prev_with_time_axis) + self.U(hidden_states)
        ))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, attention_weights
