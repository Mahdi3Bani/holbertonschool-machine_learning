#!/usr/bin/env python3
"""save and load functions"""


import tensorflow.keras as K


def save_weights(network, filename):
    """function to save the model"""
    network.save_weights(filename)


def load_weights(network, filename):
    '''function to loas a model from a filepath'''
    return K.models.load_weights(filename)
