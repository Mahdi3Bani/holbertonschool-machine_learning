#!/usr/bin/env python3
"""updates a variable using the gradient descent
with momentum optimization algorithm"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates a variable using the gradient descent
    with momentum optimization algorithm"""

    V = beta1 * v + (1 - beta1) * grad
    updated_var = var - alpha * V
    return updated_var, V
