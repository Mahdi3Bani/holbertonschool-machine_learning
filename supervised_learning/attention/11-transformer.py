#!/usr/bin/env python3
"""attention"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Transformer class"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input,
                 max_seq_target, drop_rate=0.1):
        """
        Init the class
        :param N: The number of encoder and decoder blocks
        :param dm: The model depth
        :param h: The number of heads
        :param hidden: The number of hidden units
        :param input_vocab: Size of the input vocabulary
        :param target_vocab: Size of the target vocabulary
        :param max_seq_input: Maximum sequence length for inputs
        :param max_seq_target: Maximum sequence length for targets
        :param drop_rate: The drop rate for dropout
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def encode(self, inputs, training, encoder_mask):
        """Encodes the input sequence"""
        return self.encoder(inputs, training, encoder_mask)

    def decode(self, target, enc_output, training, look_ahead_mask, decoder_mask):
        """Decodes the target sequence using the encoder's output"""
        return self.decoder(target, enc_output, training, look_ahead_mask, decoder_mask)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Call function"""
        enc_output = self.encode(inputs, training, encoder_mask)
        dec_output = self.decode(target, enc_output, training, look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)
        return final_output
