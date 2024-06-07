#!/usr/bin/env python3
"""attention"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock class"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Init the class
        :param dm: The model depth
        :param h: The number of heads
        :param hidden: The number of hidden unit
        :param drop_rate: The drop rate for dropout
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation="relu"
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def masked_mha_block(self, x, look_ahead_mask, training):
        """Masked multi-head attention block"""
        attention, _ = self.mha1(x, x, x, look_ahead_mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(attention + x)
        return out1

    def mha_block(self, x, enc_output, padding_mask, training):
        """Multi-head attention block"""
        attention, _ = self.mha2(x, enc_output, enc_output, padding_mask)
        attention = self.dropout2(attention, training=training)
        out2 = self.layernorm2(attention + x)
        return out2

    def ffn_block(self, x, training):
        """Feed forward block"""
        ffn = self.dense_hidden(x)
        ffn = self.dense_output(ffn)
        ffn = self.dropout3(ffn, training=training)
        out3 = self.layernorm3(ffn + x)
        return out3

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """call function"""
        out1 = self.masked_mha_block(x, look_ahead_mask, training)
        out2 = self.mha_block(out1, enc_output, padding_mask, training)
        out3 = self.ffn_block(out2, training)
        return out3
