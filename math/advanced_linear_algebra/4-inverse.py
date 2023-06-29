#!/usr/bin/env python3
"""inverse"""
determinant = __import__("0-determinant").determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """calculates the inverse of a matrix:

            matrix is a list of lists whose
            inverse should be calculated

            If matrix is not a list of lists, raise a
            TypeError with the message matrix must be a
            list of lists

            If matrix is not square or is empty, raise a ValueError
            with the message matrix must be a non-empty square matrix

            Returns: the inverse of matrix, or None if matrix is singular
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

    det = determinant(matrix)
    if det == 0:
        return None
    adj = adjugate(matrix)

    inverse_matrix = []

    for i in range(len(matrix)):
        inverse_row = []
        for j in range(len(matrix)):
            inverse_row.append(adj[i][j] / det)

        inverse_matrix.append(inverse_row)

    return inverse_matrix
