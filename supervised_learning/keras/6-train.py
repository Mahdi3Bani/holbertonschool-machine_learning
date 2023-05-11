#!/usr/bin/env python3
"""update the previous function and to also analyze validaiton data"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, verbose=True, shuffle=False):
    """update the previous function and
    to also analyze validaiton data"""
    callbacks = []
    if early_stopping and validation_data:
        early_stopping = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stopping)
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       early_stopping=early_stopping, patience=patience,
                       verbose=verbose, shuffle=shuffle, callbacks=callbacks)
