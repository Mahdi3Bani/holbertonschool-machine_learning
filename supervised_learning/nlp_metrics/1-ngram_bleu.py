#!/usr/bin/env python3
"""nlp metrics"""
import numpy as np


def ngram_precision(candidate_ngrams, reference_ngrams):
    """Calculate precision for n-grams"""
    clipped_counts = 0

    # Count clipped n-grams
    for ngram in candidate_ngrams:
        if ngram in reference_ngrams:
            clipped_counts += 1

    total_counts = len(candidate_ngrams)

    # Avoid division by zero
    if total_counts == 0:
        return 0

    precision = clipped_counts / total_counts
    return precision


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence"""

    BP = min(1, np.exp(1 - len(min(references, key=len)) / len(sentence)))

    candidate_ngrams = []
    for i in range(len(sentence) - (n - 1)):
        candidate_ngrams.append(tuple(sentence[i:i + n]))

    reference_ngrams = []
    for ref in references:
        for i in range(len(ref) - (n - 1)):
            reference_ngrams.append(tuple(ref[i:i + n]))

    precision = ngram_precision(candidate_ngrams, reference_ngrams)

    return BP * np.exp(np.log(precision))
