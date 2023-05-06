#!/usr/bin/env python3
"""    Calculates the cost of a neural network with L2 regularization.
"""


import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.
    """

    lambtha = 0.01
    reg_losses = []
    for var in tf.trainable_variables():
        if 'kernel' in var.name:
            reg_losses.append(tf.nn.l2_loss(var))

    l2_loss = lambtha * tf.reduce_sum(reg_losses)
    return cost + l2_loss
