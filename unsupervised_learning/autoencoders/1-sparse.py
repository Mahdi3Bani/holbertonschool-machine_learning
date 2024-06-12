#!/usr/bin/env python3
"""autoencoder"""

import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """creates a sparse autoencoder"""
    reg = K.regularizers.l1(lambtha)
    
    # Encoder
    inputs = K.Input(shape=(input_dims,))
    x = inputs
    for layer in hidden_layers:
        x = K.layers.Dense(layer, activation='relu')(x)
    latent = K.layers.Dense(latent_dims, activation='relu', activity_regularizer=reg)(x)
    
    # Decoder
    x = latent
    for layer in reversed(hidden_layers):
        x = K.layers.Dense(layer, activation='relu')(x)
    outputs = K.layers.Dense(input_dims, activation='sigmoid')(x)
    
    # Models
    encoder_model = K.Model(inputs, latent)
    decoder_model = K.Model(latent, outputs)
    autoencoder_model = K.Model(inputs, decoder_model(encoder_model(inputs)))

    # Compile the autoencoder model
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, autoencoder_model
