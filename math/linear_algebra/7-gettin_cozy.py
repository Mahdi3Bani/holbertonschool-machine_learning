#!/usr/bin/env python3
"""conctinate 2 matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenate 2 matrices"""
    if axis not in [0, 1]:
        return None

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        new_mat = []
        for i in mat1:
            new_mat.append(i.copy())
        for i in mat2:
            new_mat.append(i.copy())
        return new_mat

    else:
        if len(mat1) != len(mat2):
            return None
        new_mat = []
        for i in range(len(mat1)):
            row = mat1[i].copy() + mat2[i].copy()
            new_mat.append(row)

        return new_mat
