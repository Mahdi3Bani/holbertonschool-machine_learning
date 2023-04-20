#!/usr/bin/env python3
"""
calculates the softmax cross-entropy loss of a prediction
"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """
calculates the softmax cross-entropy loss of a prediction
    """

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (labels=y, logits=y_pred),
                          name='softmax_cross_entropy_loss')

    return loss
