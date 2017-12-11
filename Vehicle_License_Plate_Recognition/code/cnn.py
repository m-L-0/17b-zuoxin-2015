#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cnn.py
# @Author: Joker
# @Date  : 2017/12/5

import tensorflow as tf
from readTFrecord import license

train = license('train.tfrecord', 17447)
test = license('test.tfrecord', 1000)

# 一，函数声明部分

def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.9))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

# 二，定义输入输出结构

# 声明一个占位符，None表示输入图片的数量不定，48*24图片分辨率
xs = tf.placeholder(tf.float32, [None, 1152])
# 类别是0-67总共68个类别，对应输出分类结果
ys = tf.placeholder(tf.float32, [None, 68])
with tf.name_scope("reshape"):
    # x_image又把xs reshape成了48*24*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
    x_image = tf.reshape(xs, [-1, 48, 24, 1])

# 三，搭建网络,定义算法公式，也就是forward时的计算
with tf.name_scope("conv1"):
    # 第一层卷积操作
    # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
    W_conv1 = weight_variable([3, 3, 1, 48])
    # 对于每一个卷积核都有一个对应的偏置量。
    b_conv1 = bias_variable([48])
    # 图片乘以卷积核，并加上偏执量，卷积结果48x24x32
    h_conv1 = prelu(conv2d(x_image, W_conv1) + b_conv1)
with tf.name_scope("pool1"):
    # 池化结果24x12x32 卷积结果乘以池化卷积核
    h_pool1 = max_pool_2x2(h_conv1)
with tf.name_scope("conv2"):
    # 第二层卷积操作
    # 32通道卷积，卷积出64个特征
    w_conv2 = weight_variable([3, 3, 48, 96])
    # 64个偏执数据
    b_conv2 = bias_variable([96])
    # 注意h_pool1是上一层的池化结果，#卷积结果24x12x64
    h_conv2 = prelu(conv2d(h_pool1, w_conv2) + b_conv2)
with tf.name_scope("pool2"):
    # 池化结果12x6x64
    h_pool2 = max_pool_2x2(h_conv2)
    # 原图像尺寸48*24，第一轮图像缩小为24*12，共有32张，第二轮后图像缩小为12*6，共有64张
with tf.name_scope("conv3"):
    w_conv3 = weight_variable([3, 3, 96, 144])
    b_conv3 = bias_variable([144])
    h_conv3 = prelu(conv2d(h_pool2,w_conv3) + b_conv3)
with tf.name_scope("pool3"):
    h_pool3 = max_pool_2x2(h_conv3)
with tf.name_scope("fc1"):
    # 第三层全连接操作
    # 二维张量，第一个参数12*6*64的patch，也可以认为是只有一行12*6*64个数据的卷积，第二个参数代表卷积个数共1024个
    W_fc1 = weight_variable([6 * 3 * 144, 1024])
    # 1024个偏执数据
    b_fc1 = bias_variable([1024])
    # 将第二层卷积池化结果reshape成只有一行12*6*64个数据# [n_samples, 12, 6, 64] ->> [n_samples, 12*6*64]
    h_pool2_flat = tf.reshape(h_pool3, [-1, 6 * 3 * 144])
    # 卷积操作，结果是1*1*1024，单行乘以单列等于1*1矩阵，matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘，自动认为是前行向量后列向量
    h_fc1 = prelu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
with tf.name_scope("dropout"):
    keep_prob = tf.placeholder(tf.float32)
    # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
    # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
    h_fc2_drop = tf.nn.dropout(h_fc1, keep_prob)  # 对卷积结果执行dropout操作
with tf.name_scope("fc3"):
    # 第四层输出操作
    # 二维张量，1*1024矩阵卷积，共10个卷积，对应我们开始的ys长度为10
    W_fc3 = weight_variable([1024, 68])
    b_fc3 = bias_variable([68])
    # 最后的分类，结果为1*1*10 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
    # y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    y_conv=tf.matmul(h_fc2_drop, W_fc3) + b_fc3
with tf.name_scope("loss"):
    # 四，定义loss(最小误差概率)，选定优化优化loss，
    # cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))  # 定义交叉熵为loss函数
    cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv,labels=tf.argmax(ys,1)))
with tf.name_scope("Adam_optimizer"):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化
    # GradientDescentOptimizer AdamOptimizer
label_predict=tf.argmax(y_conv,1)
# 五，开始数据训练以及评测
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train_writer = tf.summary.FileWriter('graphs/')
# train_writer.add_graph(tf.get_default_graph())
saver = tf.train.Saver()
test = test.next_batch(1000)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = train.next_batch(batch_size=50)
        if i % 100 == 0:
            # print(batch[1][0])
            # print(sess.run(label_predict, feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})[0])
            train_accuracy = accuracy.eval(feed_dict={xs: test[0], ys: test[1], keep_prob: 1.0})
            print("step %d, 验证集的正确率 %g" % (i, train_accuracy))
        train_step.run(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})

    saver.save(sess, 'saved_model/lu0.97/model.ckpt')

    print("该模型的正确率为 %g" % accuracy.eval(feed_dict={xs: test[0], ys: test[1], keep_prob: 1.0}))