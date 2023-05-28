#!/usr/bin/env python3
"""Inception Block"""

import keras as K
from keras.layers import Input
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    input_shape = (224, 224, 3)
    input_layer = Input(shape=input_shape)

    # First conv layer
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7), strides=(2, 2),
                            padding='same', activation='relu')(input_layer)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2), padding='same')(conv1)

    # Second conv layer
    conv2 = K.layers.Conv2D(filters=64,
                             kernel_size=(1, 1),
                             padding="same",
                             activation="relu")(pool1)
    conv3 = K.layers.Conv2D(filters=192,
                             kernel_size=(3, 3),
                             padding="same",
                             activation="relu")(conv2)
    pool2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=(2, 2),
                                       padding="same")(conv3)

    # Inception blocks
    inception2a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception2b = inception_block(inception2a, [128, 128, 192, 32, 96, 64])

    pool3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=(2, 2),
                                       padding="same")(inception2b)

    inception4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])

    pool4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=(2, 2),
                                       padding="same")(inception4e)

    inception5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])
    # Average pooling layer
    avg_pool = K.layers.AvgPool2D(pool_size=(7, 7),
                                     padding="same")(inception5b)
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)
    # Fully connected layer
    fc = K.layers.Dense(units=1000, activation='softmax')(dropout)

    model = K.models.Model(inputs=input_layer, outputs=fc)

    return model
