#!/usr/bin/env python3
"""Converts a trained Gensim Word2Vec model to a Keras Embedding layer"""
from gensim.models import Word2Vec
from keras.layers import Embedding
import numpy as np


def gensim_to_keras(model: Word2Vec) -> Embedding:
    """
    Converts a trained Gensim Word2Vec model to a Keras Embedding layer"""
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    embedding_layer = Embedding(input_dim=vocab_size, 
                                output_dim=vector_size, 
                                weights=[weights], 
                                trainable=True)
    
    return embedding_layer
