#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_visualization.py
# @Author: Joker
# @Date  : 2017/11/14
# http://www.360doc.com/content/17/0611/21/42392246_661965445.shtml
import tensorflow as tf
import matplotlib.pyplot as plt


def data_visualization(tfrecord_filename, num):
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(
        value,
        features={
            'images': tf.FixedLenFeature([28, 28], tf.float32),
            'labels': tf.FixedLenFeature([], tf.int64)
        })
    images = features['images']
    labels = features['labels']

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    dice_label = {0: "T恤", 1: "裤子", 2: "套衫", 3: "裙子", 4: "外套", 5: "凉鞋", 6: "汗衫", 7: "运动鞋", 8: "包", 9: "踝靴"}
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num):
            imgs, labs = sess.run([images, labels])
            print('batch' + str(i+1) + ':')
            print(labs, dice_label[labs])
            plt.figure(figsize=(1, 1))
            plt.imshow(imgs, cmap='gray')
            plt.show()

        coord.request_stop()
        coord.join(threads)


data_visualization('train.tfrecord', 10)
