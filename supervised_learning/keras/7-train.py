#!/usr/bin/env python3
"""update the previous function and to train the model with learning rate decay"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """update the previous function and
    to train the model with learning rate decay"""
    callback = []
    if validation_data:
        callback.append(K.callbacks.LearningRateScheduler
                        (alpha / (1 + decay_rate * epochs)))
    if early_stopping and validation_data:
        callback.append(K.callbacks.EarlyStopping
                        (patience=patience))
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callback)
