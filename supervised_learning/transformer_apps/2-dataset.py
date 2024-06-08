#!/usr/bin/env python3
"""add wrapper for the encode instance method """


import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf


class Dataset:
    """add wrapper for the encode instance method """
    def __init__(self):
        '''init func to load the dataset'''
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', 
            split=['train', 'validation'], 
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Tokenize the dataset"""
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode the Portuguese and English sentences"""
        pt_tokens = [self.tokenizer_pt.vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy()) + \
            [self.tokenizer_pt.vocab_size+1]
        en_tokens = [self.tokenizer_en.vocab_size] + \
            self.tokenizer_en.encode(en.numpy()) + \
            [self.tokenizer_en.vocab_size+1]
        return pt_tokens, en_tokens


    def tf_encode(self, pt, en):
        """TensorFlow wrapper for the encode method"""
        pt_lang, en_lang = tf.py_function(func=self.encode,
                                          inp=[pt, en],
                                          Tout=[tf.int64, tf.int64])
        pt_lang.set_shape([None])
        en_lang.set_shape([None])
        return pt_lang, en_lang
