#!/usr/bin/env python3
"""FastText model creation and training."""
from gensim.models import FastText
from typing import List


def fasttext_model(sentences: List[List[str]], 
                   vector_size: int = 100, 
                   min_count: int = 5, 
                   window: int = 5,
                   negative: int = 5, 
                   cbow: bool = True, 
                   iterations: int = 5, 
                   seed: int = 0, 
                   workers: int = 1) -> FastText:
    """
    Initialize and train a FastText model.
    """
    sg = 0 if cbow else 1
    model = FastText(vector_size=vector_size,
                     window=window, 
                     min_count=min_count, 
                     workers=workers,
                     negative=negative, 
                     seed=seed, 
                     sg=sg)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=iterations)
    return model
