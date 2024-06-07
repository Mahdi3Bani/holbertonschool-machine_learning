usr/bin/env python3
"""Resnet50."""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Resnet50."""
    init = K.initializers.HeNormal()
    input_layer = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(64, 7, strides=2,
                            padding='same',
                            kernel_initializer=init)(input_layer)
    batch_norm = K.layers.BatchNormalization()(conv1)

    activation = K.layers.Activation('relu')(batch_norm)

    max_pooling = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2,
                                        padding='same')(activation)

    # ------------- convolution --------------------#
    projection2 = projection_block(max_pooling, [64, 64, 256], s=1)
    identity2 = identity_block(projection2, [64, 64, 256])
    identity3 = identity_block(identity2, [64, 64, 256])
    projection2 = projection_block(identity3, [128, 128, 512])

    identity4 = identity_block(projection2, [128, 128, 512])
    identity5 = identity_block(identity4, [128, 128, 512])
    identity6 = identity_block(identity5, [128, 128, 512])

    projection3 = projection_block(identity6, [256, 256, 1024])
    identity7 = identity_block(projection3, [256, 256, 1024])
    identity8 = identity_block(identity7, [256, 256, 1024])
    identity9 = identity_block(identity8, [256, 256, 1024])
    identity10 = identity_block(identity9, [256, 256, 1024])
    identity11 = identity_block(identity10, [256, 256, 1024])

    projection4 = projection_block(identity11, [512, 512, 2048])
    identity12 = identity_block(projection4, [512, 512, 2048])
    identity13 = identity_block(identity12, [512, 512, 2048])

    avg = K.layers.AveragePooling2D(pool_size=(7, 7)
                                    )(identity13)

    output_layer = K.layers.Dense(1000,
                                  kernel_initializer=init,
                                  activation='softmax')(avg)

    model = K.Model(input_layer, output_layer)
    return model
