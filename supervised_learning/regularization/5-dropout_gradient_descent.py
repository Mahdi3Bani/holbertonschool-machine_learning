import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Update the weights of a neural network with Dropout regularization using gradient descent.

    Arguments:
    Y -- one-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data
    classes -- the number of classes
    m -- the number of data points
    weights -- dictionary of the weights and biases of the neural network
    cache -- dictionary of the outputs and dropout masks of each layer of the neural network
    alpha -- learning rate
    keep_prob -- probability that a node will be kept
    L -- number of layers of the network

    Returns:
    Nothing. The weights of the network are updated in place.
    """

    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    A_prev = cache[f"A{L-1}"]

    for layer in range(L, 0, -1):

        dW = np.dot(dZ, cache['A' + str(layer - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = np.matmul(weights['W' + str(layer)].T, dZ)

        A = cache["A" + str(layer - 1)]

        if layer > 1:
            dZ = dZ *\
                (1 - np.power(A, 2)) * \
                (cache['D' + str(layer - 1)] / keep_prob)

        weights[f"W{layer}"] = weights[f"W{layer}"] - alpha * dW
        weights[f"b{layer}"] = weights[f"b{layer}"] - alpha * db
