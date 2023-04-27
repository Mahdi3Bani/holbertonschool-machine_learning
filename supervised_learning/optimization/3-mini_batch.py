#!/usr/bin/env python3
""" Train a loaded neural network model using mini-batch gradient descent """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """ train a loaded neural network model using mini-batch gradient descent """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + ".meta")
        loader.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        train_feed_dict = {x: X_train, y: Y_train}
        valid_feed_dict = {x: X_valid, y: Y_valid}
        m = X_train.shape[0]
        steps_per_epoch = m // batch_size
        if m % batch_size != 0:
            steps_per_epoch += 1
        for i in range(epochs):
            X_train_shuffled, Y_train_shuffled = shuffle_data(X_train, Y_train)
            train_cost = sess.run(loss, feed_dict=train_feed_dict)
            train_accuracy = sess.run(accuracy, feed_dict=train_feed_dict)
            valid_cost = sess.run(loss, feed_dict=valid_feed_dict)
            valid_accuracy = sess.run(accuracy, feed_dict=valid_feed_dict)
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            for j in range(steps_per_epoch):
                start = j * batch_size
                end = start + batch_size
                if end > m:
                    end = m
                X_batch = X_train_shuffled[start:end]
                Y_batch = Y_train_shuffled[start:end]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                if j % 100 == 0:
                    step_cost = sess.run(
                        loss, feed_dict={x: X_batch, y: Y_batch})
                    step_accuracy = sess.run(accuracy, feed_dict={
                                             x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(j))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
        save_path = loader.save(sess, save_path)
    return save_path
