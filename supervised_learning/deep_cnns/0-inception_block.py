#!/usr/bin/env python3
"""Inception Block"""

import keras as K


def inception_block(A_prev, filters):
    """Write a function def inception_block(A_prev, filters):
    that builds an inception block as described in Going Deeper with
    Convolutions (2014):

        A_prev is the output from the previous layer

        filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
        respectively:

            *F1 is the number of filters in the 1x1 convolution
            *F3R is the number of filters in the 1x1 convolution
            before the 3x3 convolution
            *F3 is the number of filters in the 3x3 convolution
            *F5R is the number of filters in the 1x1 convolution
            before the 5x5 convolution
            *F5 is the number of filters in the 5x5 convolution
            *FPP is the number of filters in the 1x1 convolution
            after the max pooling

        All convolutions inside the inception block should use a
        rectified linear activation (ReLU)

        Returns: the concatenated output of the inception block
"""
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch with constant filter F1
    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                              padding='same', activation='relu')(A_prev)

    # Rest of the code remains the same
    conv3x3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                                     padding='same', activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                              padding='same', activation='relu')(conv3x3_reduce)

    conv5x5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                                     padding='same', activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5),
                              padding='same', activation='relu')(conv5x5_reduce)

    pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    pool_proj = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                padding='same', activation='relu')(pool)

    concat = K.layers.concatenate(
        [conv1x1, conv3x3, conv5x5, pool_proj], axis=-1)

    return concat
