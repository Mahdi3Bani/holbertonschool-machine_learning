#!/usr/bin/env python3
"""adjugate matrix of a matrix"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Write a function def adjugate(matrix):
    that calculates the adjugate matrix of a matrix:

        matrix is a list of lists whose adjugate
        matrix should be calculated

        If matrix is not a list of lists, raise a
        TypeError with the message matrix must be a
        list of lists

        If matrix is not square or is empty, raise a
        ValueError with the message matrix must be a
        non-empty square matrix

    Returns: the adjugate matrix of matrix
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
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]

    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []
    for i in range(len(matrix)):
        adjugate_row = []
        for j in range(len(matrix[0])):
            adjugate_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix
