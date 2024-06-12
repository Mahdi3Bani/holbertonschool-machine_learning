#!/usr/bin/env python3
"""autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""

    # Encoder
    inputs = K.Input(shape=input_dims)
    x = inputs
    for conv in filters:
        x = K.layers.Conv2D(conv, (3, 3), activation='relu',
                            padding='same')(x)
        x = K.layers.MaxPooling2D((2, 2), padding='same')(x)
    latent = x

    # Decoder
    x = latent
    for filter in reversed(filters[1:]):
        x = K.layers.Conv2D(
            filters=filter,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )(x)
        x = K.layers.UpSampling2D((2, 2))(x)
    x = K.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
    )(x)
    x = K.layers.UpSampling2D(size=(2, 2))(x)
    outputs = K.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="sigmoid",
    )(x)

    # Models
    encoder = K.Model(inputs, latent)
    decoder_input = K.Input(shape=latent_dims)
    decoder_output = decoder_input
    for filter in reversed(filters[1:]):
        decoder_output = K.layers.Conv2D(
            filters=filter,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )(decoder_output)
        decoder_output = K.layers.UpSampling2D((2, 2))(decoder_output)
    decoder_output = K.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
    )(decoder_output)
    decoder_output = K.layers.UpSampling2D(size=(2, 2))(decoder_output)
    decoder_output = K.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="sigmoid",
    )(decoder_output)
    decoder = K.Model(decoder_input, decoder_output)

    autoencoder_output = decoder(encoder(inputs))
    autoencoder = K.Model(inputs, autoencoder_output)

    # Compile the autoencoder model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
