#!/usr/bin/env python3
"""Full DenseNet-121."""


from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """DenseNet-121 architecture using Keras."""
    init = K.initializers.HeNormal()
    inputs = K.Input(shape=(224, 224, 3))

    batch_norm1 = K.layers.BatchNormalization()(inputs)
    activation1 = K.layers.Activation('relu')(batch_norm1)
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same', kernel_initializer=init)(activation1)
    max_pool1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding='same')(conv1)

    dense_block1, num_filters1 = dense_block(max_pool1, 64, growth_rate, 6)
    transition_layer1, num_filters1 = transition_layer(dense_block1,
                                                       num_filters1, compression)

    dense_block2, num_filters2 = dense_block(transition_layer1, num_filters1,
                                             growth_rate, 12)
    transition_layer2, num_filters2 = transition_layer(dense_block2,
                                                       num_filters2, compression)

    dense_block3, num_filters3 = dense_block(transition_layer2, num_filters2,
                                             growth_rate, 24)
    transition_layer3, num_filters3 = transition_layer(dense_block3,
                                                       num_filters3, compression)

    dense_block4, _ = dense_block(transition_layer3,
                                  num_filters3, growth_rate, 16)

    global_avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7))(dense_block4)
    output_layer = K.layers.Dense(units=1000, activation='softmax',
                                  kernel_initializer=init)(global_avg_pool)

    model = K.Model(inputs=inputs, outputs=output_layer)
    return model
