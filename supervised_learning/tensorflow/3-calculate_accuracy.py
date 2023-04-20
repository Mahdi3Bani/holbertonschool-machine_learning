#!/usr/bin/env python3
"""
Calculates the accuracy of a prediction
"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction
    """
    y_pred_labels = tf.argmax(y_pred, axis=1)

    y_true_labels = tf.argmax(y, axis=1)

    correct_predictions = tf.reduce_sum(
        tf.cast(tf.equal(y_pred_labels, y_true_labels), tf.float32))

    accuracy = tf.divide(correct_predictions,
                         tf.cast(tf.shape(y)[0], tf.float32))

    return accuracy
