#!/usr/bin/env python3
"""word2vec"""


from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim word2vec model
    """
    model_type = 'cbow' if cbow else 'skip-gram'
    print(f"Training Word2Vec model using {model_type} method...")
    
    model = Word2Vec(sentences,
                     size=size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=0 if cbow else 1,
                     iter=iterations,
                     seed=seed,
                     workers=workers)
    
    print("Word2Vec model trained successfully.")
    return model
