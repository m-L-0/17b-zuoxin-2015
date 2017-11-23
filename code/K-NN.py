#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : KNN.py
# @Author: Joker
# @Date  : 2017/11/15

import tensorflow as tf
import numpy as np
from read_tfrecord import read_tfrecord


def loadmnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets('data/fashion', one_hot=True)


def knn(data):
    train_x, train_y = data.train.next_batch(5000)
    test_x, test_y = data.test.next_batch(200)
    #二值化处理，消除颜色深度的影响，然而并没有提高多少精确度
    # for i in range(len(train_x)):
    #     for j in range(len(train_x[0])):
    #         if train_x[i][j] >0:
    #             train_x[i][j]=1.0
    # for i in range(len(test_x)):
    #     for j in range(len(test_x[0])):
    #         if test_x[i][j] > 0:
    #             test_x[i][j] = 1.0

    xtr = tf.placeholder(tf.float32, [None, 784])
    xte = tf.placeholder(tf.float32, [784])
    distance = tf.pow(tf.reduce_sum(tf.pow(tf.add(xtr, tf.negative(xte)), 2), reduction_indices=1), 1/2)

    pred = tf.argmin(distance, 0)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    right = 0
    for i in range(200):
        ansindex = sess.run(pred, {xtr: train_x, xte: test_x[i, :]})
        print('prediction is ', np.argmax(train_y[ansindex]), 'true value is ', np.argmax(test_y[i]))
        if np.argmax(test_y[i]) == np.argmax(train_y[ansindex]):
            right += 1.0
    accracy = right / 200.0
    print(accracy)


if __name__ == "__main__":
    mnist = loadmnist()
    knn(mnist)
