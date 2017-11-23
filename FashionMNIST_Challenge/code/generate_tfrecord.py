#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : fashion-mnist.py
# @Author: Joker
# @Date  : 2017/11/15
import tensorflow as tf


def loadmnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets('data/fashion', validation_size=5000)


def generate_tfrecord(label, data):
    with tf.python_io.TFRecordWriter(label+'.tfrecord') as writer:
        list_data = [[a, b] for a, b in zip(data.images, data.labels)]
        for i in list_data:
            images = tf.train.Feature(float_list=tf.train.FloatList(value=i[0]))
            labels = tf.train.Feature(int64_list=tf.train.Int64List(value=[i[1]]))
            features = tf.train.Features(feature={'labels': labels, 'images': images})
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    mnist = loadmnist()
    generate_tfrecord('train', mnist.train)
    generate_tfrecord('validation', mnist.validation)
    generate_tfrecord('test', mnist.test)
