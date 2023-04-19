#!/usr/bin/env python3
"""Shape of matrix"""


def matrix_shape(matrix):
    """funtion that count shape"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return (shape)
