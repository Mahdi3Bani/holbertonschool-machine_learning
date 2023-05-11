#!/usr/bin/env python3
"""update the previous function and to also analyze validaiton data"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, verbose=True, shuffle=False):
    """update the previous function and
    to also analyze validaiton data"""
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       early_stopping=early_stopping,patience=patience,
                       verbose=verbose, shuffle=shuffle)
