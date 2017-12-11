#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : readTFrecord.py
# @Author: Joker
# @Date  : 2017/12/5
import tensorflow as tf
import numpy as np

class license(object):

    def __init__(self, tfrecord_fileename, num):
        filename_queue = tf.train.string_input_producer([tfrecord_fileename])
        reader = tf.TFRecordReader()
        key, value = reader.read(filename_queue)
        features = tf.parse_single_example(
            value,
            features={
                'images': tf.FixedLenFeature([1, 1152], tf.int64),
                'labels': tf.FixedLenFeature([], tf.int64)
            })
        images = features['images']
        labels = features['labels']
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self._images = np.zeros((num, 1152))
        self._labels = np.zeros((num, 68))
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(num):
                imgs, labs = sess.run([images, labels])
                self._images[i] = imgs
                if labs == 52 | labs == 48:
                    self._labels[i][labs] = 0.97
                else:
                    self._labels[i][labs] = 1.0

            coord.request_stop()
            coord.join(threads)

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = num

    def next_batch(self, batch_size):
        start = self._index_in_epoch

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples-start
            images_rest_part = self._images[start:self._num_examples,:]
            labels_rest_part = self._labels[start:self._num_examples,:]

            start = 0

            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end,:]
            labels_new_part = self._labels[start:end,:]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end,:], self._labels[start:end,:]