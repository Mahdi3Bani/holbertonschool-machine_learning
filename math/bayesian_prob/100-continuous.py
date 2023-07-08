#!/usr/bin/env python3
'''Likelihood'''

import numpy as np


def comb(n, k):
    """comb function"""
    if not 0 <= k <= n:
        return 0
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


def likelihood(x, n, P):
    """comment"""

    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    likelihoods = np.array(
        [comb(n, x) * p**x * (1-p)**(n-x) for p in P])

    return likelihoods


def intersection(x, n, P, Pr):
    '''comment'''

    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.ndim != P.ndim:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return likelihood(x, n, P) * Pr


def marginal(x, n, P, Pr):
    """comment"""
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.ndim != P.ndim:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, P, Pr):
    """comment"""
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.ndim != P.ndim:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)

from scipy import special


def posterior(x, n, p1, p2):
    """comment"""

    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p1 >= p2:
        raise ValueError("p2 must be greater than p1")
    b1 = special.btdtr(x + 1, n - x + 1, p1)
    b2 = special.btdtr(x + 1, n - x + 1, p2)
    return b2 - b1
