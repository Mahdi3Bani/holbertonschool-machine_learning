#!/usr/bin/env python3
"""builds a modified version of the
LeNet-5 architecture using tensorflow:
"""


import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the
    LeNet-5 architecture using tensorflow:

    x: is a tf.placeholder of shape (m, 28, 28, 1) 
    containing the input images for the network
        *m is the number of images

    y: is a tf.placeholder of shape (m, 10)
    containing the one-hot labels for the network

    The model should consist of the following layers in order:
        -Convolutional layer with 6 kernels of shape 5x5 with same padding
        -Max pooling layer with kernels of shape 2x2 with 2x2 strides
        -Convolutional layer with 16 kernels of shape 5x5 with valid padding
        -Max pooling layer with kernels of shape 2x2 with 2x2 strides
        -Fully connected layer with 120 nodes
        -Fully connected layer with 84 nodes
        -Fully connected softmax output layer with 10 nodes

    All layers requiring initialization should initialize 
    their kernels with the he_normal initialization method:
    tf.contrib.layers.variance_scaling_initializer()

    All hidden layers requiring activation should use 
    he relu activation function

    you may import tensorflow as tf

    you may NOT use tf.keras

    Returns:
        -a tensor for the softmax activated output
        -a training operation that utilizes Adam optimization (with default hyperparameters)
        -a tensor for the loss of the netowrk
        -a tensor for the accuracy of the network
    """

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # Flatten the previous output
    flatten = tf.layers.flatten(pool2)

    # Fully connected layer with 120 nodes
    fc1 = tf.layers.dense(flatten, units=120, activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # Fully connected layer with 84 nodes
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # Fully connected softmax output layer with 10 nodes
    logits = tf.layers.dense(fc2, units=10,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # Softmax activated output
    softmax_output = tf.nn.softmax(logits)

    # Loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(softmax_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Training operation
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return softmax_output, train_op, loss, accuracy
