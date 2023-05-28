#!/usr/bin/env python3
"""Identity Block"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    F11, F3, F12 = filters

    # Shortcut branch
    shortcut = A_prev

    # First layer
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
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

    # Add shortcut to the main path
    add = K.layers.Add()([bn3, shortcut])
    activated_output = K.layers.Activation('relu')(add)

    return activated_output
