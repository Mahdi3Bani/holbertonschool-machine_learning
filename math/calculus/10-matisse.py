#!/usr/bin/env python3
"""dervate a function"""


def poly_derivative(poly):
    """dereivta a poly function"""

    if not isinstance(poly, list) or not poly or len(poly) == 0:
        return None
    new_list = []
    for i in range(1, len(poly)):
        new_list.append(poly[i] * i)

    if len(new_list) == 0:
        return [0]
    return new_list
