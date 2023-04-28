#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    l = []
    l.append(data[0])
    bias = 1
    for i in range(1, len(data)):
        l.append(beta * l[i - 1] + (1 - beta) * data[i])
        bias *= beta
        l[i] = l[i]/(1 - bias)
    return l
