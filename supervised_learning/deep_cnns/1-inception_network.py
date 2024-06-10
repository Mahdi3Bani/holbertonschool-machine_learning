#!/usr/bin/env python3
"""Inception Block"""


from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Write a function def inception_network():
    that builds the inception network as described in Going
    Deeper with Convolutions (2014):

        You can assume the input data will have shape (224, 224, 3)
        All convolutions inside and outside the inception block should
        use a rectified linear activation (ReLU)
        Returns: the keras model"""
    input_layer = K.layers.Input(shape=(224, 224, 3))

    # First conv layer
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            activation='relu')(input_layer)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(conv1)

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
    inception3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])

    pool3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding="same")(inception3b)

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
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding="valid")(inception5b)
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)
    # Fully connected layer
    fc = K.layers.Dense(units=1000, activation='softmax')(dropout)

    model = K.models.Model(inputs=input_layer, outputs=fc)

    return model
