#!/usr/bin/env python3
"""Updates a variable using the RMSProp optimization algorithm."""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Updates a variable using the RMSProp optimization algorithm."""

    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)

