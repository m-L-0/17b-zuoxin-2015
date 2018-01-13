#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cnn.py
# @Author: Joker
# @Date  : 2017/12/13

import tensorflow as tf
import numpy as np
from readTfrecord import train_batch, valid_batch, test_batch

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def prelu_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def prelu(_x, _p, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _p = tf.cast(_p, tf.float32)
        return tf.maximum(0.0, _x) + _p * tf.minimum(0.0, _x)

xs, ys = tf.train.batch(train_batch(), batch_size=50)
x1s, y1s = tf.train.batch(valid_batch(), batch_size=200)

with tf.name_scope("reshape"):
    x_image = tf.reshape(xs, [-1, 36, 48, 3])
    y_shape = tf.reshape(ys, [-1, 4, 11])
    y_lab1 = tf.reshape(tf.split(y_shape, 4, 1)[0], [-1, 11])
    y_lab2 = tf.reshape(tf.split(y_shape, 4, 1)[1], [-1, 11])
    y_lab3 = tf.reshape(tf.split(y_shape, 4, 1)[2], [-1, 11])
    y_lab4 = tf.reshape(tf.split(y_shape, 4, 1)[3], [-1, 11])
    x1_image = tf.reshape(x1s, [-1, 36, 48, 3])
    y1_shape = tf.reshape(y1s, [-1, 4, 11])
    y1_lab1 = tf.reshape(tf.split(y1_shape, 4, 1)[0], [-1, 11])
    y1_lab2 = tf.reshape(tf.split(y1_shape, 4, 1)[1], [-1, 11])
    y1_lab3 = tf.reshape(tf.split(y1_shape, 4, 1)[2], [-1, 11])
    y1_lab4 = tf.reshape(tf.split(y1_shape, 4, 1)[3], [-1, 11])


with tf.name_scope("conv1"):
    w_conv1 = weight_variable([3, 3, 3, 48])
    b_conv1 = bias_variable([48])
    p_conv1 = prelu_variable([48])
    h_conv1 = prelu(conv2d(x_image, w_conv1) + b_conv1, p_conv1)
    h1_conv1 = prelu(conv2d(x1_image, w_conv1) + b_conv1, p_conv1)

with tf.name_scope("pool1"):
    # 18 24
    h_pool1 = max_pool_2x2(h_conv1)
    h1_pool1 = max_pool_2x2(h1_conv1)

with tf.name_scope("conv2"):
    w_conv2 = weight_variable([3, 3, 48, 96])
    b_conv2 = bias_variable([96])
    p_conv2 = prelu_variable([96])
    h_conv2 = prelu(conv2d(h_pool1, w_conv2) + b_conv2, p_conv2)
    h1_conv2 = prelu(conv2d(h1_pool1, w_conv2) + b_conv2, p_conv2)

with tf.name_scope("pool2"):
    # 9 12
    h_pool2 = max_pool_2x2(h_conv2)
    h1_pool2 = max_pool_2x2(h1_conv2)

with tf.name_scope("conv3"):
    w_conv3 = weight_variable([3, 3, 96, 144])
    b_conv3 = bias_variable([144])
    p_conv3 = prelu_variable([144])
    h_conv3 = prelu(conv2d(h_pool2, w_conv3) + b_conv3, p_conv3)
    h1_conv3 = prelu(conv2d(h1_pool2, w_conv3) + b_conv3, p_conv3)

with tf.name_scope("pool3"):
    # 5 6
    h_pool3 = max_pool_2x2(h_conv3)
    h1_pool3 = max_pool_2x2(h1_conv3)

with tf.name_scope("fc1"):
    w_fc1 = weight_variable([5*6*144, 1024])
    b_fc1 = bias_variable([1024])
    p_fc1 = prelu_variable([1024])
    h_pool2_flat = tf.reshape(h_pool3, [-1, 5*6*144])
    h_fc1 = prelu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, p_fc1)

    h1_pool2_flat = tf.reshape(h1_pool3, [-1, 5 * 6 * 144])
    h1_fc1 = prelu(tf.matmul(h1_pool2_flat, w_fc1) + b_fc1, p_fc1)

with tf.name_scope("dropout"):
    keep_prob = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc1, keep_prob)
    h1_fc2_drop = tf.nn.dropout(h1_fc1, keep_prob)

with tf.name_scope("out1"):
    w_out1 = weight_variable([1024, 11])
    b_out1 = bias_variable([11])
    y_out1 = tf.matmul(h_fc2_drop, w_out1) + b_out1
    y1_out1 = tf.matmul(h1_fc2_drop, w_out1) + b_out1

with tf.name_scope("out2"):
    w_out2 = weight_variable([1024, 11])
    b_out2 = bias_variable([11])
    y_out2 = tf.matmul(h_fc2_drop, w_out2) + b_out2
    y1_out2 = tf.matmul(h1_fc2_drop, w_out2) + b_out2

with tf.name_scope("out3"):
    w_out3 = weight_variable([1024, 11])
    b_out3 = bias_variable([11])
    y_out3 = tf.matmul(h_fc2_drop, w_out3) + b_out3
    y1_out3 = tf.matmul(h1_fc2_drop, w_out3) + b_out3

with tf.name_scope("out4"):
    w_out4 = weight_variable([1024, 11])
    b_out4 = bias_variable([11])
    y_out4 = tf.matmul(h_fc2_drop, w_out4) + b_out4
    y1_out4 = tf.matmul(h1_fc2_drop, w_out4) + b_out4

with tf.name_scope("loss1"):
    cross_entropy1 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out1, labels=tf.argmax(y_lab1, 1)))

with tf.name_scope("loss2"):
    cross_entropy2 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out2, labels=tf.argmax(y_lab2, 1)))

with tf.name_scope("loss3"):
    cross_entropy3 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out3, labels=tf.argmax(y_lab3, 1)))

with tf.name_scope("loss4"):
    cross_entropy4 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out4, labels=tf.argmax(y_lab4, 1)))

with tf.name_scope("loss"):
    cross_entropy = cross_entropy1 + cross_entropy2 + cross_entropy3 + cross_entropy4

with tf.name_scope("Adam_optimizer"):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    l1=tf.argmax(y1_lab1, 1);p1=tf.argmax(y1_out1, 1)
    correct_prediction1 = tf.equal(tf.argmax(y1_out1, 1), tf.argmax(y1_lab1, 1))
    correct_prediction2 = tf.equal(tf.argmax(y1_out2, 1), tf.argmax(y1_lab2, 1))
    correct_prediction3 = tf.equal(tf.argmax(y1_out3, 1), tf.argmax(y1_lab3, 1))
    correct_prediction4 = tf.equal(tf.argmax(y1_out4, 1), tf.argmax(y1_lab4, 1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
    accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))
    y1_tmp1 = tf.concat([y1_out1, y1_out2], 1)
    y1_tmp2 = tf.concat([y1_out3, y1_out4], 1)
    y1_out = tf.concat([y1_tmp1, y1_tmp2], 1)
    accuracy = tf.reduce_mean(tf.cast(tf.cast(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.reshape(y1_out, [-1, 4, 11]), 2),
                                                                              tf.argmax(y1_shape, 2)), tf.float32),1),tf.int64),tf.float32))

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(30000):
        if i % 100 == 0:
            [cc,valid_accuracy, v1, v2, v3, v4] = sess.run([cross_entropy,accuracy, accuracy1, accuracy2, accuracy3, accuracy4],
                                                           feed_dict={keep_prob: 1.0} )
            print("step %d, 验证集的正确率 %g 第一位的正确率 %g 第二位的正确率 %g 第三位的正确率 %g 第四位的正确率 %g 交叉熵代价 %g"
                  % (i, valid_accuracy, v1, v2, v3, v4, cc))
        train_step.run(feed_dict={keep_prob: 0.5})
    saver.save(sess, 'saved_model/second/model.ckpt')
    coord.request_stop()
    coord.join(threads)