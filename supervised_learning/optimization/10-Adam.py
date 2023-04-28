#!/usr/bin/env python3
"""updates a variable in place using the Adam optimization algorithm."""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm"""

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1, beta2=beta2,
                                       epsilon=epsilon)
    return optimizer.minimize(loss)
