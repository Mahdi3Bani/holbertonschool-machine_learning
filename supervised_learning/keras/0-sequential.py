#!/usr/bin/env python3
"""builds a neural network with the Keras library"""


import tensorflow.keras as K
"""from tensorflow import keras"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    model = K.models.Sequential()

    model.add(K.layers.Dense(layers[0],
                             activation=activations[0], input_shape=(nx,),
                             kernel_regularizer=K.regularizers.l2(lambtha)))

    for i in range(1, len(layers)):
        model.add(K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lambtha)))
        if keep_prob < 1:
            model.add(K.layers.Dropout(1-keep_prob))
    return model
