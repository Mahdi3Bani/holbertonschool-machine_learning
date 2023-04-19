#!/usr/bin/env python3
"""calculate the integral of  function"""


def poly_integral(poly, C=0):
    """dereivta a poly function"""

    if (not isinstance(poly, list) or not poly
            or len(poly) == 0
            or not isinstance(C, int)):
        return None
    if poly == [0]:
        if C:
            return [C]
        else:
            return [0]
    new_list = [C]
    for i in range(len(poly)):
        if (poly[i] / (i + 1)).is_integer():
            new_list.append(int(poly[i] / (i + 1)))
        else:
            new_list.append(poly[i] / (i + 1))

    return (new_list)
