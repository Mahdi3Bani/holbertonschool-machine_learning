#!/usr/bin/env python3
'''creates an autoencoder'''


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Builds an autoencoder model
    """

    encoder_input = keras.layers.Input(shape=(input_dims,))

    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    encoder_output = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.models.Model(encoder_input, encoder_output, name='encoder')

    decoder_input = keras.layers.Input(shape=(latent_dims,))

    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.models.Model(decoder_input, decoder_output, name='decoder')

    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = keras.models.Model(autoencoder_input, decoded, name='autoencoder')

    auto.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy())

    return encoder, decoder, auto
