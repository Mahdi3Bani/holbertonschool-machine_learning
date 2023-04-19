#!/usr/bin/env python3
"""matrix transpose"""


def matrix_transpose(matrix):
    """function that create a transpose"""
    row_numb = len(matrix)
    col_numb = len(matrix[0])

    transposed = []
    for col in range(col_numb):
        row_list = []
        for row in range(row_numb):
            row_list.append(0)
        transposed.append(row_list)

    for row in range(row_numb):
        for col in range(col_numb):
            transposed[col][row] = matrix[row][col]

    return (transposed)
