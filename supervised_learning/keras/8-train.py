#!/usr/bin/env python3
"""update the previous function and to
save the best iteration of the model"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """update the previous function and
    to save the best iteration of the model"""
    def lr_decay(epochs):
        return alpha / (1 + decay_rate * epochs)

    callback = []
    if early_stopping and validation_data:
        callback.append(K.callbacks.EarlyStopping(patience=patience))

    if save_best:
        callback.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                    save_best_only=True,
                                                    monitor='val_loss',
                                                    mode='min'))
    if learning_rate_decay and validation_data:
        callback.append(K.callbacks.LearningRateScheduler(
            lr_decay,
            verbose=True
        ))

    if early_stopping and validation_data:
        callback.append(K.callbacks.EarlyStopping(patience=patience))

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callback
    )
