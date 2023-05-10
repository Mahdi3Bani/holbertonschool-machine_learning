#!/usr/bin/env python3
"""Builds a neural network using Keras"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras

    """

    X_input = K.Input(shape=(nx,))

    regularization = K.regularizers.l2(lambtha)

    x = K.layers.Dense(layers[0], activation=activations[0], kernel_regularizer=regularization)(inputs)
    x = K.layers.Dropout(1 - keep_prob)(x)


    for i in range(len(layers)):
        X = x = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regularization)(x)

        
        X = K.layers.Dropout(1 - keep_prob)(X)

    model = K.Model(inputs=X_input, outputs=X)

    return model
