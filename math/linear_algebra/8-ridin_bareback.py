#!/usr/bin/env python3
"""matix mult"""


def mat_mul(mat1, mat2):
    """matrix mult"""
    if len(mat1[0]) != len(mat2):
        return None

    new_mat = []

    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            row.append(0)
        new_mat.append(row)

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                new_mat[i][j] += mat1[i][k] * mat2[k][j]

    return new_mat
