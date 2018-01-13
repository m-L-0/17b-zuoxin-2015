#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : storeTfrecord.py
# @Author: Joker
# @Date  : 2017/12/12

import tensorflow as tf
import csv
import matplotlib.image as mpimg
from scipy import misc
import numpy as np

list_labels = []
list_images = []
csvfile = open('captcha/labels/labels.csv', 'r')
reader = csv.reader(csvfile)

for n, j in reader:
    if len(str(j)) < 4:
        tmp_y = str(j) + (4-len(str(j))) * '$'
    else:
        tmp_y = str(j)
    print(tmp_y)
    out_y = np.zeros([4, 11])
    for i in range(4):
        if tmp_y[i] == "$":
            tmp = 10
        else:
            tmp = int(tmp_y[i])
        out_y[i][tmp] = 1
    list_labels.append(out_y)

for i in range(40000):
    pic = mpimg.imread('captcha/images/' + str(i) + '.jpg')
    tmppic = misc.imresize(pic, (36, 48))
    list_images.append(tmppic)

list_images = np.array(list_images)
list_labels = np.array(list_labels)

for i in range(10):
    with tf.python_io.TFRecordWriter('tfrecord 1-hot/data'+str(i)+'.tfrecord') as writer:
        for j in range(i*4000, (i+1)*4000):
            img = list_images[j].reshape(5184)
            images = tf.train.Feature(int64_list=tf.train.Int64List(value=img))
            print(list_labels[j])
            lab = list_labels[j].reshape(44)
            labels = tf.train.Feature(float_list=tf.train.FloatList(value=lab))
            features = tf.train.Features(feature={'labels': labels, 'images': images})
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

# 输出验证集与测试集的各个数量
#
# dict_len1 = {1: 0, 2: 0, 3: 0, 4: 0}
# dict_len2 = {1: 0, 2: 0, 3: 0, 4: 0}
# for i in range(32000, 36000):
#     dict_len1[len(str(list_labels[i]))] += 1
# for i in range(36000, 40000):
#     dict_len2[len(str(list_labels[i]))] += 1
#
# print(dict_len1)
# print(dict_len2)
