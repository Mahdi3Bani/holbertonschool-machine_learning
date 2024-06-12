#!/usr/bin/env python3
'''creates an autoencoder'''


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Builds an autoencoder model
    """

    encoder_input = layers.Input(shape=(input_dims,))

    x = encoder_input
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)

    encoder_output = layers.Dense(latent_dims, activation='relu')(x)

    encoder = models.Model(encoder_input, encoder_output, name='encoder')

    decoder_input = layers.Input(shape=(latent_dims,))

    x = decoder_input
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation='relu')(x)

    decoder_output = layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = models.Model(decoder_input, decoder_output, name='decoder')

    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = models.Model(autoencoder_input, decoded, name='autoencoder')

    auto.compile(optimizer=optimizers.Adam(), loss=losses.BinaryCrossentropy())

    return encoder, decoder, auto
