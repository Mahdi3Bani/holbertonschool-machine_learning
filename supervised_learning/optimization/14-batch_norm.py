#!/usr/bin/env python3
"""Creates a batch normalization layer for a neural network in TensorFlow."""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.
    """
    dense = tf.layers.Dense(units=n,
                            kernel_initializer=tf.contrib.layers.
                            variance_scaling_initializer(
        mode="FAN_AVG"), use_bias=False)

    mean, variance = tf.nn.moments(dense(prev), axes=0)

    gamma = tf.Variable(tf.ones([n]), dtype=tf.float32)
    beta = tf.Variable(tf.zeros([n]), dtype=tf.float32)

    bn = tf.nn.batch_normalization(dense(
        prev), mean=mean, variance=variance, offset=beta,
        scale=gamma, variance_epsilon=1e-8)

    output = activation(bn)

    return output
