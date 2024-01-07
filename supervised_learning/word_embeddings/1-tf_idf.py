#!/usr/bin/env python3

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(sentences, vocab=None):
    if vocab is None:
        vocab_set = set()
        for sentence in sentences:
            words = sentence.split()

        for word in words:
            vocab_set.add(word)


    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features