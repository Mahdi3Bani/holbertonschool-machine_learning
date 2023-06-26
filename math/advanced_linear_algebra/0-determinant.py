#!/usr/bin/env python3
"""determinant"""


def determinant(matrix):
    """that calculates the determinant of a matrix:

        matrix is a list of lists whose determinant should
        be calculated

        If matrix is not a list of lists, raise a TypeError
        with the message matrix must be a list of lists

        If matrix is not square, raise a ValueError with the
        message matrix must be a square matrix

        The list [[]] represents a 0x0 matrix

        Returns: the determinant of matrix
    """
    if not isinstance(matrix, list) or len (matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for mat in matrix:
        if not isinstance(mat, list):
            raise TypeError("matrix must be a list of lists")

    rows = len(matrix)
    cols = len(matrix[0])

    if rows != cols:
        ValueError("matrix must be a square matrix")
    
    if matrix == [[]]:
        return 1
    
    if len(matrix) == 1:
        return matrix[0][0]
    
    if rows == 2:
        det = matrix[0][0] * matrix[1][1]  - matrix[1][0] * matrix[0][1]
        return det
    else:
        det = 0
        sign = 1
        for i in range(cols):
            mat = []
            for row in matrix[1:]:
                new_row = []
                for j in range(len(row)):
                    if j != 1:
                        new_row.append(row[j])
                mat.append(new_row)
        det += sign * matrix[0][i] * determinant(mat)
        sign *= -1
    return det
   


