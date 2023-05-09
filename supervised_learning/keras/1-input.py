#!/usr/bin/env python3
"""Builds a neural network using Keras"""


import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras

    """

    X_input = k.Input(shape=(nx,))

    X = X_input

    for i in range(len(layers)):
        X = k.layers.Dense(units=layers[i],
                           kernel_regularizer=k.regularizers.l2(lambtha))(X)
        X = k.layers.Activation(activations[i])(X)
        X = k.layers.Dropout(1 - keep_prob)(X)

    model = k.Model(inputs=X_input, outputs=X)

    return model
