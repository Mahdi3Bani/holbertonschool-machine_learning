#!/usr/bin/env python3
"""comment"""

import numpy as np


def initialize(X, k):
    '''comment'''
    if k <= 0:
        return None
    
    n, d = X.shape
    if n < k:
        return None
    
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    
    centroids = np.random.uniform(low=min_values, high=max_values, size=(k, d))
    return centroids
