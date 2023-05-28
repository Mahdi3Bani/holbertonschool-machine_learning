#!/usr/bin/env python3
"""Transition Layer
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Write a function def transition_layer(X, nb_filters, compression):
    that builds a transition layer as described in Densely Connected
    Convolutional Networks:

        *X is the output from the previous layer
        *nb_filters is an integer representing the number of filters in X
        *compression is the compression factor for the transition layer
        *Your code should implement compression as used in DenseNet-C
        *All weights should use he normal initialization
        *All convolutions should be preceded by Batch Normalization
        and a rectified linear activation (ReLU), respectively

        Returns: The output of the transition layer and the
        number of filters within the output, respectively
"""
    bn = K.layers.BatchNormalization(axis=3)(X)
    relu = K.layers.Activation('relu')(bn)
    conv = K.layers.Conv2D(filters=int(nb_filters * compression),
                           kernel_size=(1, 1), padding='same',
                           kernel_initializer='he_normal')(relu)
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv)

    return avg_pool, int(nb_filters * compression)
