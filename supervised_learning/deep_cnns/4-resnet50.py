#!/usr/bin/env python3
"""Projection Block
"""


from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Write a function def resnet50():
    that builds the ResNet-50 architecture as
    described in Deep Residual Learning for Image Recognition (2015):

        *You can assume the input data will have shape (224, 224, 3)
        *All convolutions inside and outside the blocks should be
        followed by batch normalization along the channels axis and a
        rectified linear activation (ReLU), respectively.
        *All weights should use he normal initialization
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same',
                            kernel_initializer=init)(X)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    activation = K.layers.Activation('relu')(bn1)
    max_pool = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(activation)
    conv2_0 = projection_block(max_pool, [64, 64, 256], s=1)
    conv2_1 = identity_block(conv2_0, [64, 64, 256])
    conv2_2 = identity_block(conv2_1, [64, 64, 256])
    conv3_0 = projection_block(conv2_2, [128, 128, 512])
    conv3_1 = identity_block(conv3_0, [128, 128, 512])
    conv3_2 = identity_block(conv3_1, [128, 128, 512])
    conv3_3 = identity_block(conv3_2, [128, 128, 512])
    conv4_0 = projection_block(conv3_3, [256, 256, 1024])
    conv4_1 = identity_block(conv4_0, [256, 256, 1024])
    conv4_2 = identity_block(conv4_1, [256, 256, 1024])
    conv4_3 = identity_block(conv4_2, [256, 256, 1024])
    conv4_4 = identity_block(conv4_3, [256, 256, 1024])
    conv4_5 = identity_block(conv4_4, [256, 256, 1024])
    conv5_0 = projection_block(conv4_5, [512, 512, 2048])
    conv5_1 = identity_block(conv5_0, [512, 512, 2048])
    conv5_2 = identity_block(conv5_1, [512, 512, 2048])
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1))(conv5_2)
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=init)(avg_pool)

    return K.Model(inputs=X, outputs=dense)
    