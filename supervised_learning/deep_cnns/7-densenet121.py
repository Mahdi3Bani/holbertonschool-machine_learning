#!/usr/bin/env python3
""" Inception Block """


from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ DenseNet-121 architecture """
    input = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    x = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=init)(input)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x, nb_filters = dense_block(x, 64, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    x = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    x = K.layers.Flatten()(x)

    output = K.layers.Dense(units=1000, activation='softmax', kernel_initializer=init)(x)

    return K.Model(inputs=input, outputs=output)
