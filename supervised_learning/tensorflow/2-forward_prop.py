#!/usr/bin/env python3
"""creates the forward propagation graph for the neural network"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    """
    for i in range(len(layer_sizes)):
        layer_size = layer_size[i]
        activation = activations[i]
        layer = create_layer(input_layer, layer_size, activation)
        input_layer = layer
    return input_layer
