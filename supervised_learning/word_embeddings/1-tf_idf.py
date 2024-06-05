#!/usr/bin/env python3
"""TF-IDF"""


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding"""
    if vocab is None:
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out()
    return embeddings, features
