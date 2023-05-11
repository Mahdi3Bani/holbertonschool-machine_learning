#!/usr/bin/env python3
"""Builds a neural network using Keras"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras

    """

    X = Y = K.Input(shape=(nx,))

    regularization = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        dense = (K.layers.
                 Dense(layers[i],
                       activation=activations[i],
                       kernel_regularizer=regularization))
        Y = dense(Y)

        if i < len(layers) - 1:
            dropout = K.layers.Dropout(1 - keep_prob)

            y = dropout(Y)

    return K.Model(inputs=X, outputs=Y)
