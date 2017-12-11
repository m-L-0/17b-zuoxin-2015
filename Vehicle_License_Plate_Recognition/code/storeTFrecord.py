#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : storeTFrecord.py
# @Author: Joker
# @Date  : 2017/12/4

import os
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import misc
from transform_label import label

root = os.pardir+os.sep+"car"+os.sep
list_image = []
list_label = []
#三通道转为一通道灰度图
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#图片二值化并且统一成24*48
def pic(file):
    ppic = mpimg.imread(file)
    ppic = rgb2gray(ppic)
    tmppic = misc.imresize(ppic, (48, 24))
    mean = np.mean(tmppic)
    for i in range(tmppic.shape[0]):
        for j in range(tmppic.shape[1]):
            if tmppic[i][j] > mean:
                tmppic[i][j] = 1
            else:
                tmppic[i][j] = 0
    return tmppic

for rt, dirs, files in os.walk(root):
    for f in files:
        p, lab = os.path.split(rt)
        img = pic(rt + "\\" + f).reshape(1152)
        biaoqian = label[lab]
        list_image.append(img)
        list_label.append(biaoqian)
print(len(list_label))

list_data = [[a, b] for a, b in zip(list_image, list_label)]
random.shuffle(list_data)
#
# with tf.python_io.TFRecordWriter('train.tfrecord') as writer:
#     for i in range(17447):
#         images = tf.train.Feature(int64_list=tf.train.Int64List(value=list_data[i][0]))
#         print(list_data[i][1])
#         labels = tf.train.Feature(int64_list=tf.train.Int64List(value=[list_data[i][1]]))
#         features = tf.train.Features(feature={'labels': labels, 'images': images})
#         example = tf.train.Example(features=features)
#         writer.write(example.SerializeToString())
#
#
# with tf.python_io.TFRecordWriter('test.tfrecord') as writer:
#     for i in range(17447,18447):
#         images = tf.train.Feature(int64_list=tf.train.Int64List(value=list_data[i][0]))
#         print(list_data[i][1])
#         labels = tf.train.Feature(int64_list=tf.train.Int64List(value=[list_data[i][1]]))
#         features = tf.train.Features(feature={'labels': labels, 'images': images})
#         example = tf.train.Example(features=features)
#         writer.write(example.SerializeToString())

with tf.python_io.TFRecordWriter('test.tfrecord') as writer:
    for i in range(5428):
        images = tf.train.Feature(int64_list=tf.train.Int64List(value=list_data[i][0]))
        print(list_data[i][1])
        labels = tf.train.Feature(int64_list=tf.train.Int64List(value=[list_data[i][1]]))
        features = tf.train.Features(feature={'labels': labels, 'images': images})
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
