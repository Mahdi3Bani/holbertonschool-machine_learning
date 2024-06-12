#!/usr/bin/env python3
"""nlp metrics"""
import numpy as np
from typing import List


def uni_bleu(references: List[List[str]], sentence: List[str]) -> float:
    """Calculate the unigram BLEU score"""
    candidate_length = len(sentence)
    reference_length = min(len(ref) for ref in references)

    if candidate_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - reference_length / candidate_length)

    # Count clipped n-gram matches
    clipped_count = 0
    total_count = len(sentence)
    for word in set(sentence):
        max_reference_count = max(ref.count(word) for ref in references)
        clipped_count += min(sentence.count(word), max_reference_count)

    if total_count == 0:
        return 0.0
    precision = clipped_count / total_count

    bleu_score = brevity_penalty * precision

    return bleu_score
