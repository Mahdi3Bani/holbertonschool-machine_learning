#!/usr/bin/env python3
'''rnn decoder class'''

import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.initializers import glorot_uniform


class RNNDecoder(tf.keras.layers.Layer):
    '''RNN decoder'''
    def __init__(self, vocab, embedding, units, batch):
        """init function"""
        super(RNNDecoder, self).__init__()
        self.vocab_size = vocab
        self.embedding_dim = embedding
        self.units = units
        self.batch = batch

        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.gru = GRU(units=self.units, return_sequences=True, return_state=True,
                       recurrent_initializer=glorot_uniform())
        self.F = Dense(units=self.vocab_size)


    def call(self, x, s_prev, hidden_states):
        '''create an embedding vector with embedding layers
        then pass embedding vector to GRU layer'''
        x = self.embedding(x)
        attention = SelfAttention(self.units)
        context, weights = attention(s_prev, hidden_states)

        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        output = self.F(output)

        return output, state
