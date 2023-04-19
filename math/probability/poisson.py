#!/usr/bin/env python3
"""class that represents a poisson distribution"""


class Poisson:
    """class that represents a poisson distribution"""
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
            self.lambtha = sum(data) / float(len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""

        k = int(k)
        e = 2.7182818285
        if k < 0:
            return 0
        else:
            fact = 1
            for i in range(1, k+1):
                fact = fact * i
            return ((self.lambtha ** k) * (e ** (-self.lambtha))) / fact

    def cdf(self, k):
        '''calculates thr value of the cdf for a giving number of success'''
        k = int(k)
        if k < 0:
            return 0
        else:
            cdf = 0.0
            for i in range(k+1):
                cdf = cdf + self.pmf(i)
            return cdf
