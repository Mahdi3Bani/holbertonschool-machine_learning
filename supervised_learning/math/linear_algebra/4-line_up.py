#!/usr/bin/env python3
"""add 2 array"""


def add_arrays(arr1, arr2):
    """add function"""

    if len(arr1) != len(arr2):
        return None

    arr3 = []
    for i in range(len(arr1)):
        arr3.append(arr1[i] + arr2[i])
    return (arr3)
