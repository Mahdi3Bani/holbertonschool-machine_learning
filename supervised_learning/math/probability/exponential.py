#!/usr/bin/env python3
"""class that represents a distribution exponential"""


class Exponential:
    """represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:

                raise ValueError('data must contain multiple values')
            self.lambtha = float(len(data)) / sum(data)

    def pdf(self, x):
        """repsent exp pdf"""
        e = 2.7182818285
        if x < 0:
            return 0
        else:
            return self.lambtha * e ** ((-self.lambtha) * x)

    def cdf(self, x):
        """repesent exp cdf"""
        e = 2.7182818285
        if x < 0:
            return 0
        else:
            return 1 - (e ** ((-self.lambtha) * x))
