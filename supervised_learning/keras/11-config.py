#!/usr/bin/env python3
"""save and load functions"""


import tensorflow.keras as K


def save_config(network, filename):
    """function to save the model"""
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    '''function to loas a model from a filepath'''
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
