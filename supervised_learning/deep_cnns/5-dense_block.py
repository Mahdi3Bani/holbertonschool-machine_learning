#!/usr/bin/env python3
"""Dense Block
"""


from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Write a function def dense_block(X, nb_filters, growth_rate, layers):
    that builds a dense block as described in Densely Connected Convolutional
    Networks:

        *X is the output from the previous layer
        *nb_filters is an integer representing the number
        of filters in X
        *growth_rate is the growth rate for the dense block
        *layers is the number of layers in the dense block
        *You should use the bottleneck layers used for DenseNet-B
        *All weights should use he normal initialization
        *All convolutions should be preceded by Batch Normalization
        and a rectified linear activation (ReLU), respectively

        Returns: The concatenated output of each layer within
        the Dense Block and the number of filters within the
        concatenated outputs, respectively
"""
    concat_output = X
    num_filters = nb_filters

    for _ in range(layers):
        # Bottleneck layer
        bn1 = K.layers.BatchNormalization(axis=3)(concat_output)
        relu1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=(1, 1), padding='same',
                                kernel_initializer='he_normal')(relu1)

        # Convolution layer
        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        relu2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3), padding='same',
                                kernel_initializer='he_normal')(relu2)

        # Concatenate the output of each layer
        concat_output = K.layers.Concatenate(axis=3)([concat_output, conv2])

        # Update the number of filters
        num_filters += growth_rate

    return concat_output, num_filters
