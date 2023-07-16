#!/usr/bin/env python3
"""comment"""

import numpy as np


def variance(X, C):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    _, d = X.shape

    if d != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, None] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    var = np.sum((X - C[clss]) ** 2)

    return var
