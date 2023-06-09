#!/usr/bin/env python3
"""
evaluates the output of a neural network:
"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    evaluates the output of a neural network:
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        y_pred = graph.get_tensor_by_name('y_pred:0')
        accuracy = graph.get_collection('accuracy')[0]
        loss = graph.get_collection('loss')[0]

        feed_dict = {x: X, y: Y}
        y_pred_val, accuracy_val, loss_val = sess.run(
            [y_pred, accuracy, loss], feed_dict=feed_dict)

    return y_pred_val, accuracy_val, loss_val
