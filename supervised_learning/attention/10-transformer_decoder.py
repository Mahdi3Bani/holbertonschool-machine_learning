#!/usr/bin/env python3
"""attention"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder class"""

    def __init__(self, N, dm, h, hidden,
                 target_vocab, max_seq_len, drop_rate=0.1):
        """
        Init the class
        :param N: The number of decoder blocks
        :param dm: The model depth
        :param h: The number of heads
        :param hidden: The number of hidden units
        :param target_vocab: Size of the target vocabulary
        :param max_seq_len: Maximum sequence length
        :param drop_rate: The drop rate for dropout
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def embed_and_scale(self, x):
        """Embed and scale the input tensor"""
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        return x

    def add_positional_encoding(self, x):
        """Add positional encoding to the input tensor"""
        seq_len = tf.shape(x)[1]
        x += self.positional_encoding[:seq_len]
        return x

    def apply_decoder_blocks(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Apply decoder blocks"""
        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask, padding_mask)
        return x

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Call function"""
        x = self.embed_and_scale(x)
        x = self.add_positional_encoding(x)
        x = self.dropout(x, training=training)
        x = self.apply_decoder_blocks(x, encoder_output, training, look_ahead_mask, padding_mask)
        return x
