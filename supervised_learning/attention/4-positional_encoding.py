#!/usr/bin/env python3
"""positioanl encoding function"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """positional encodin"""
    positional_encoding_matrix = np.zeros((max_seq_len, dm))

    for pos in range(max_seq_len):
        for i in range(dm//2):
            positional_encoding_matrix[pos, 2*i] = np.sin(
                pos / (10000 ** ((2 * i) / dm)))
            positional_encoding_matrix[pos, 2*i + 1] = np.cos(
                pos / (10000 ** ((2 * i) / dm)))

    return positional_encoding_matrix
