#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    moving_average_list = []
    f = 0
    for i in range(len(data)):
        f = beta * f + (1 - beta) * data[i]
        moving_average_list.append(f / (1 - beta ** (i + 1)))
    return moving_average_list
