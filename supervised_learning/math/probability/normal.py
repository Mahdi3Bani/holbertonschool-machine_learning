#!/usr/bin/env python3
"""class that represents a normal distribution"""


class Normal:
    """class that represent a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        self.data = data
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(self.data, list):
                raise TypeError("data must be a list")
            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(self.data) / len(self.data))

            self.stddev = float((sum(
                [(x - self.mean) ** 2 for x in self.data]
                ) / len(self.data))**0.5)

    def z_score(self, x):
        """function calculates the z score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """function calculates the x values"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """calculates the pdf value"""
        e = 2.7182818285
        pi = 3.1415926536
        return (1 / (self.stddev * (2 * pi) ** 0.5)) *\
            e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)

    def cdf(self, x):
        """calculates the cdf values"""
        pi = 3.1415926536
        return 0.5 * (1 + self._erf((x - self.mean) /
                                    (self.stddev * (2 ** 0.5))))

    def _erf(self, x):
        """function erf"""
        pi = 3.1415926536
        return ((2 / pi ** 0.5) * (
            x - (x ** 3 / 3) +
            (x ** 5 / 10) -
            (x ** 7 / 42) +
            (x ** 9 / 216)
        ))
