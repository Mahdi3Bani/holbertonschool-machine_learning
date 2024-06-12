#!/usr/bin/env python3
"""autoencoder"""
import tensorflow.keras as K

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder model.
    """
    
    # Encoder
    encoder_input = K.Input(shape=(input_dims,))
    x = encoder_input
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)
    latent_output = K.layers.Dense(latent_dims, activation='relu')(x)
    encoder = K.Model(encoder_input, latent_output, name='encoder')
    
    # Decoder
    decoder_input = K.Input(shape=(latent_dims,))
    x = decoder_input
    for units in reversed(hidden_layers):
        x = K.layers.Dense(units, activation='relu')(x)
    decoder_output = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(decoder_input, decoder_output, name='decoder')
    
    # Autoencoder
    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = K.Model(autoencoder_input, decoded, name='autoencoder')
    
    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, autoencoder
