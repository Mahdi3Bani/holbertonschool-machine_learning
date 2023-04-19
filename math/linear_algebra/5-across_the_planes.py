#!/usr/bin/env python3
"""add 2d array"""


def add_matrices2D(mat1, mat2):
    """add function"""

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    mat3 = []
    for i in range(len(mat1)):
        new_l = []
        for j in range(len(mat1[i])):
            new_l.append(mat1[i][j] + mat2[i][j])
        mat3.append(new_l)
    return (mat3)
