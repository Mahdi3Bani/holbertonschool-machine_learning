#!/usr/bin/env python3
"""minor matrix of a matrix"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """calculates the minor matrix of a matrix:

        matrix is a list of lists whose minor matrix
        should be calculated

        If matrix is not a list of lists, raise a
        TypeError with the message matrix must be a
        list of lists

        If matrix is not square or is empty, raise a
        ValueError with the message matrix must be a
        non-empty square matrix

        Returns: the minor matrix of matrix
    """

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]) or len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    for i in matrix:
        if len(i) != len(matrix) or len(matrix) == 0:

            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    minors_mat = []
    for i in range(len(matrix)):
        minors_row = []
        for j in range(len(matrix[i])):
            sub_matrix = [row[:j] + row[j + 1:]
                          for row in matrix[:i] + matrix[i + 1:]]
            det = determinant(sub_matrix)
            minors_row.append(det)
        minors_mat.append(minors_row)

    return minors_mat
