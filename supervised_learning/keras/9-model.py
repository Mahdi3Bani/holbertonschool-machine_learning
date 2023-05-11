#!/usr/bin/env python3
"""save and load functions"""


import tensorflow.keras as K


def save_model(network, filename):
    """function to save the model"""
    network.save(filename)


def load_model(filename):
    '''function to loas a model from a filepath'''
    return K.models.load_model(filename)
