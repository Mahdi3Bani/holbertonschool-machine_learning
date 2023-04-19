#!/usr/bin/env python3
"""class that represents a binominal distribution"""


class Binomial:
    """class that represents a binominal distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            self.n = int(n)
            if self.n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = (sum([(x - mean) ** 2 for x in data]) / len(data))
            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n

    @staticmethod
    def comb(n, k):
        """comb function"""
        if not 0 <= k <= n:
            return 0
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return (self.comb(self.n, k) *
                self.p ** k * (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        k = int(k)
        if k < 0:
            return 0
        elif k >= self.n:
            return 1
        else:
            return sum(self.pmf(i) for i in range(k + 1))
