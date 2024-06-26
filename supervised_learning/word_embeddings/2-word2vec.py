#!/usr/bin/env python3
"""Word2Vec model creation and training."""


from gensim.models import Word2Vec
from typing import List


def word2vec_model(sentences: List[List[str]], 
                   vector_size: int = 100, 
                   min_count: int = 5, 
                   window: int = 5,
                   negative: int = 5, 
                   cbow: bool = True, 
                   iterations: int = 5, 
                   seed: int = 0, 
                   workers: int = 1) -> Word2Vec:
    """
    Initialize and train a Word2Vec model.
    """
    sg = 0 if cbow else 1
    model = Word2Vec(vector_size=vector_size,
                     window=window, 
                     min_count=min_count, 
                     workers=workers,
                     negative=negative, 
                     seed=seed, 
                     sg=sg)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=iterations)
    return model
