#!/usr/bin/env python3
"""creating bag of words"""


import numpy as np
import re


def preprocess_sentence(sentence):
    """preprcoess sentece"""
    processed_sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
    words = re.findall(r'\w+', processed_sentence)
    return words


def build_vocabulary(sentences):
    """
    Build vocabulary from preprocessed sentences.
    """
    all_words = set()
    for sentence in sentences:
        for word in sentence:
            all_words.add(word)
    return sorted(all_words)


def create_word_index(vocab):
    """
    Create a dictionary mapping words to their indices in the vocabulary
    """
    word_to_index = {}
    for idx, word in enumerate(vocab):
        word_to_index[word] = idx
    return word_to_index


def count_words(sentences, word_to_index):
    """
    Count the occurrences of each word in each sentence
    """
    num_sentences = len(sentences)
    num_features = len(word_to_index)
    embeddings = np.zeros((num_sentences, num_features), dtype=int)

    for i, sentence in enumerate(sentences):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings


def bag_of_words(sentences, vocab=None):
    """
    Create a bag of words embedding matrix.
    """
    preprocessed_sentences = []
    for sentence in sentences:
        preprocessed_sentences.append(preprocess_sentence(sentence))

    if vocab is None:
        vocab = build_vocabulary(preprocessed_sentences)

    word_to_index = create_word_index(vocab)

    embeddings = count_words(preprocessed_sentences, word_to_index)

    return embeddings, vocab
