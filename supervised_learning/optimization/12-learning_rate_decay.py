#!/usr/bin/env python3
"""updates the learning rate using inverse time decay in tensorflow"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates the learning rate using inverse time decay in tensorflow"""
    return tf.train.inverse_time_decay(
        learning_rate=alpha, global_step=global_step,
        decay_steps=decay_step, decay_rate=decay_rate, staircase=True)
