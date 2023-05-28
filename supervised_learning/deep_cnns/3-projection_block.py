#!/usr/bin/env python3
"""Projection Block
"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Write a function def projection_block(A_prev, filters, s=2):
    that builds a projection block as described in Deep Residual
    Learning for Image Recognition (2015):

        *A_prev is the output from the previous layer
        *filters is a tuple or list containing
        F11, F3, F12, respectively:

            -F11 is the number of filters in the
            first 1x1 convolution
            -F3 is the number of filters in the 3x3
            convolution
            -F12 is the number of filters in the second
            1x1 convolution as well as the 1x1 convolution
            in the shortcut connection
            -s is the stride of the first convolution
            in both the main path and the shortcut connection
        *All convolutions inside the block should be
        followed by batch normalization along the channels
        axis and a rectified linear activation (ReLU), respectively.
        *All weights should use he normal initialization

    Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters

    # Shortcut branch
    shortcut = A_prev

    # Main path
    # First layer
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                            strides=(s, s),
                            padding='same',
                            kernel_initializer='he_normal')(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(bn1)

    # Second layer
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                            kernel_initializer='he_normal')(relu1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation('relu')(bn2)

    # Third layer
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                            kernel_initializer='he_normal')(relu2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Shortcut path
    shortcut = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                               strides=(s, s),
                               padding='same',
                               kernel_initializer='he_normal')(shortcut)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add shortcut to the main path
    add = K.layers.Add()([bn3, shortcut])
    activated_output = K.layers.Activation('relu')(add)

    return activated_output
