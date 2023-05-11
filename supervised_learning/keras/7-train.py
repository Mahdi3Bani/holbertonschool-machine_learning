#!/usr/bin/env python3
"""update the previous function and to train the model with learning rate decay"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0,learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """update the previous function and
    to train the model with learning rate decay"""
    callbacks = []
    if validation_data:
        if learning_rate_decay:
            decay = K.callbacks.LearningRateScheduler(lambda epoch: alpha/ (1 + decay_rate * epoch))
            callbacks.append(decay)
        if early_stopping:
            early_stopping = K.callbacks.EarlyStopping(
            patience=patience
        )
        callbacks.append(early_stopping)
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
