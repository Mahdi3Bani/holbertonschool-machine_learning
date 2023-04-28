#!/usr/bin/env python3

import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
    prev (tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation: The activation function that should be used on the output of the layer.

    Returns:
    A tensor of the activated output for the layer.
    """
    # Create a dense layer with the desired number of nodes
    dense = tf.layers.Dense(units=n, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), use_bias=False)
    
    # Compute the mean and variance of the layer inputs
    mean, variance = tf.nn.moments(dense(prev), axes=0)

    # Create variables for the scale and shift parameters
    gamma = tf.Variable(tf.ones([n]), dtype=tf.float32)
    beta = tf.Variable(tf.zeros([n]), dtype=tf.float32)

    # Create a batch normalization layer using the computed mean and variance, and the scale and shift parameters
    bn = tf.nn.batch_normalization(dense(prev), mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=1e-8)

    # Apply the activation function to the normalized layer output
    output = activation(bn)

    return output
