#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : readTfrecord.py
# @Author: Joker
# @Date  : 2017/12/14

import tensorflow as tf

train_tfrecord = ['tfrecord 1-hot/data0.tfrecord', 'tfrecord 1-hot/data1.tfrecord', 'tfrecord 1-hot/data2.tfrecord',
                  'tfrecord 1-hot/data3.tfrecord', 'tfrecord 1-hot/data4.tfrecord', 'tfrecord 1-hot/data5.tfrecord',
                  'tfrecord 1-hot/data6.tfrecord', 'tfrecord 1-hot/data7.tfrecord']
valid_tfrecord = ['tfrecord 1-hot/data8.tfrecord']
test_tfrecord = ['tfrecord 1-hot/data9.tfrecord']
train_queue = tf.train.string_input_producer(train_tfrecord, shuffle=False)
valid_queue = tf.train.string_input_producer(valid_tfrecord, shuffle=False)
test_queue = tf.train.string_input_producer(test_tfrecord, shuffle=False)


def train_batch():
    reader = tf.TFRecordReader()
    key, value = reader.read(train_queue)
    features = tf.parse_single_example(
                value,
                features={
                    'images': tf.FixedLenFeature([1, 5184], tf.int64),
                    'labels': tf.FixedLenFeature([44], tf.float32)
                })
    train_img = features['images']
    train_lab = features['labels']
    train_img = tf.cast(train_img, tf.float32)
    return [train_img,train_lab]


def valid_batch():
    reader = tf.TFRecordReader()
    key, value = reader.read(valid_queue)
    features = tf.parse_single_example(
        value,
        features={
            'images': tf.FixedLenFeature([1, 5184], tf.int64),
            'labels': tf.FixedLenFeature([44], tf.float32)
        })
    valid_img = features['images']
    valid_lab = features['labels']
    valid_img = tf.cast(valid_img, tf.float32)
    return [valid_img, valid_lab]


def test_batch():
    reader = tf.TFRecordReader()
    key, value = reader.read(test_queue)
    features = tf.parse_single_example(
        value,
        features={
            'images': tf.FixedLenFeature([1, 5184], tf.int64),
            'labels': tf.FixedLenFeature([44], tf.float32)
        })
    test_img = features['images']
    test_lab = features['labels']
    test_img = tf.cast(test_img, tf.float32)
    return [test_img, test_lab]

# a,b = test_batch()
# example_batch, label_batch = tf.train.batch(
#       [a,b], batch_size=50)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#     for i in range(50):
#         m,n=sess.run([example_batch,label_batch])
#         print(n)
#     coord.request_stop()
#     coord.join(threads)