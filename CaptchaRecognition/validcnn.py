#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cnn.py
# @Author: Joker
# @Date  : 2017/12/13

import tensorflow as tf
import numpy as np

def cnn(ppic):
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


    with tf.name_scope("reshape"):
        x1_image = tf.constant(ppic, dtype=tf.float32, shape=[1, 36, 48, 3])
        # x1_image = tf.reshape(x, [-1, 36, 48, 3])

    with tf.name_scope("conv1"):
        w_conv1 = weight_variable([3, 3, 3, 48])
        b_conv1 = bias_variable([48])
        p_conv1 = prelu_variable([48])
        h1_conv1 = prelu(conv2d(x1_image, w_conv1) + b_conv1, p_conv1)

    with tf.name_scope("pool1"):
        # 18 24
        h1_pool1 = max_pool_2x2(h1_conv1)

    with tf.name_scope("conv2"):
        w_conv2 = weight_variable([3, 3, 48, 96])
        b_conv2 = bias_variable([96])
        p_conv2 = prelu_variable([96])
        h1_conv2 = prelu(conv2d(h1_pool1, w_conv2) + b_conv2, p_conv2)

    with tf.name_scope("pool2"):
        # 9 12
        h1_pool2 = max_pool_2x2(h1_conv2)

    with tf.name_scope("conv3"):
        w_conv3 = weight_variable([3, 3, 96, 144])
        b_conv3 = bias_variable([144])
        p_conv3 = prelu_variable([144])
        h1_conv3 = prelu(conv2d(h1_pool2, w_conv3) + b_conv3, p_conv3)

    with tf.name_scope("pool3"):
        # 5 6
        h1_pool3 = max_pool_2x2(h1_conv3)

    with tf.name_scope("fc1"):
        w_fc1 = weight_variable([5*6*144, 1024])
        b_fc1 = bias_variable([1024])
        p_fc1 = prelu_variable([1024])

        h1_pool2_flat = tf.reshape(h1_pool3, [-1, 5 * 6 * 144])
        h1_fc1 = prelu(tf.matmul(h1_pool2_flat, w_fc1) + b_fc1, p_fc1)

    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        h1_fc2_drop = tf.nn.dropout(h1_fc1, keep_prob)

    with tf.name_scope("out1"):
        w_out1 = weight_variable([1024, 11])
        b_out1 = bias_variable([11])
        y1_out1 = tf.matmul(h1_fc2_drop, w_out1) + b_out1
        y1 = tf.argmax(y1_out1, 1)

    with tf.name_scope("out2"):
        w_out2 = weight_variable([1024, 11])
        b_out2 = bias_variable([11])
        y1_out2 = tf.matmul(h1_fc2_drop, w_out2) + b_out2
        y2 = tf.argmax(y1_out2, 1)

    with tf.name_scope("out3"):
        w_out3 = weight_variable([1024, 11])
        b_out3 = bias_variable([11])
        y1_out3 = tf.matmul(h1_fc2_drop, w_out3) + b_out3
        y3 = tf.argmax(y1_out3, 1)

    with tf.name_scope("out4"):
        w_out4 = weight_variable([1024, 11])
        b_out4 = bias_variable([11])
        y1_out4 = tf.matmul(h1_fc2_drop, w_out4) + b_out4
        y4 = tf.argmax(y1_out4, 1)

    with tf.Session() as sess:
        # ckpt = tf.train.latest_checkpoint('saved_model/second')
        # if ckpt:
        #     saver.restore(sess=sess, save_path=ckpt)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('./saved_model/final')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        [yo1, yo2, yo3, yo4] = sess.run([y1, y2, y3, y4], feed_dict={keep_prob: 1.0} )
        print(yo1, yo2, yo3, yo4)

        coord.request_stop()
        coord.join(threads)
        if yo1[0] == 10:
            yo1=' '
        if yo2[0] == 10:
            yo2=' '
        if yo3[0] == 10:
            yo3=' '
        if yo4[0] == 10:
            yo4=' '
        out = str(yo1[0])+str(yo2[0])+str(yo3[0])+str(yo4[0])
    return out

# if __name__ == '__main__':
#     import matplotlib.image as mpimg
#     from scipy import misc
#     pic = mpimg.imread('1.jpg')
#     tmppic = misc.imresize(pic, (36, 48))
#     out = cnn(tmppic)
#     print(out)