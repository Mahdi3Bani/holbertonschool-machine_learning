#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    l = []
    f = 0
    for i in range(len(data)):
        f = beta * f + (1 - beta) * data[i]
        l.append(f / (1 - beta ** (i + 1)))
    return l
