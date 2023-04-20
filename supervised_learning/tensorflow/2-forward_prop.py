#!/usr/bin/env python3
"""creates the forward propagation graph for the neural network"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network:
    x is the placeholder for the input data
    layer_sizes is a list containing the number of nodes in each layer of the network
    activations is a list containing the activation functions for each layer of the network
    Returns: the prediction of the network in tensor form
    """
    input_layer = x
    for i in range(len(layer_sizes)):
        layer_size = layer_size[i]
        activation = activations[i]
        layer = create_layer(input_layer, layer_size, activation)
        input_layer = layer
    return input_layer
