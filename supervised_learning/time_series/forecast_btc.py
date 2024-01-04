#!/usr/bin/env python3
"""
creates, trains, and validates a keras model for the forecasting of BTC
"""
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import numpy as np

data = np.load('preprocessed_data.npz')
X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']

time_steps = X_train.shape[1]
X_train = X_train.reshape(X_train.shape[0], time_steps, 1)
X_test = X_test.reshape(X_test.shape[0], time_steps, 1)

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)).batch(batch_size)

model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(time_steps, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(train_dataset, epochs=10, validation_data=test_dataset)

loss = model.evaluate(test_dataset)
print(f'MSE : {loss}')
