#!/usr/bin/env python3
import numpy as np
"""

"""


def positional_encoding(max_seq_len, dm):
    """positional encodin"""
    positional_encoding_matrix = np.zeros((max_seq_len, dm))

    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            positional_encoding_matrix[pos, i] = np.sin(pos / (10000 ** ((2 * i) / dm)))
            positional_encoding_matrix[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / dm)))

    return positional_encoding_matrix
