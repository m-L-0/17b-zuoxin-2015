#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : t-sne.py
# @Author: Joker
# @Date  : 2017/11/15


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def loadmnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets('data/fashion')
data = loadmnist()
# learning_rate = 100.0, early_exaggeration = 100.0, n_iter = 1000
mnist = data.train.next_batch(2000)
X_embedded = TSNE(n_components=3).fit_transform(mnist[0])
type0_x = []
type0_y = []
type0_z = []
type1_x = []
type1_y = []
type1_z = []
type2_x = []
type2_y = []
type2_z = []
type3_x = []
type3_y = []
type3_z = []
type4_x = []
type4_y = []
type4_z = []
type5_x = []
type5_y = []
type5_z = []
type6_x = []
type6_y = []
type6_z = []
type7_x = []
type7_y = []
type7_z = []
type8_x = []
type8_y = []
type8_z = []
type9_x = []
type9_y = []
type9_z = []

for i in range(len(X_embedded)):
    if mnist[1][i] == 0:
        type0_x.append(X_embedded[i][0])
        type0_y.append(X_embedded[i][1])
        type0_z.append(X_embedded[i][2])
    if mnist[1][i] == 1:
        type1_x.append(X_embedded[i][0])
        type1_y.append(X_embedded[i][1])
        type1_z.append(X_embedded[i][2])
    if mnist[1][i] == 2:
        type2_x.append(X_embedded[i][0])
        type2_y.append(X_embedded[i][1])
        type2_z.append(X_embedded[i][2])
    if mnist[1][i] == 3:
        type3_x.append(X_embedded[i][0])
        type3_y.append(X_embedded[i][1])
        type3_z.append(X_embedded[i][2])
    if mnist[1][i] == 4:
        type4_x.append(X_embedded[i][0])
        type4_y.append(X_embedded[i][1])
        type4_z.append(X_embedded[i][2])
    if mnist[1][i] == 5:
        type5_x.append(X_embedded[i][0])
        type5_y.append(X_embedded[i][1])
        type5_z.append(X_embedded[i][2])
    if mnist[1][i] == 6:
        type6_x.append(X_embedded[i][0])
        type6_y.append(X_embedded[i][1])
        type6_z.append(X_embedded[i][2])
    if mnist[1][i] == 7:
        type7_x.append(X_embedded[i][0])
        type7_y.append(X_embedded[i][1])
        type7_z.append(X_embedded[i][2])
    if mnist[1][i] == 8:
        type8_x.append(X_embedded[i][0])
        type8_y.append(X_embedded[i][1])
        type8_z.append(X_embedded[i][2])
    if mnist[1][i] == 9:
        type9_x.append(X_embedded[i][0])
        type9_y.append(X_embedded[i][1])
        type9_z.append(X_embedded[i][2])

fig = plt.figure()
ax = Axes3D(fig)

#将数据点分成三部分画，在颜色上有区分度
type0 = ax.scatter(type0_x,type0_y,type0_z,c='b') #绘制数据点
type1 = ax.scatter(type1_x,type1_y,type1_z,c='y')
type2 = ax.scatter(type2_x,type2_y,type2_z,c='r')
type3 = ax.scatter(type3_x,type3_y,type3_z,c='g')
type4 = ax.scatter(type4_x,type4_y,type4_z,c='k')
type5 = ax.scatter(type5_x,type5_y,type5_z,c='c')
type6 = ax.scatter(type6_x,type6_y,type6_z,c='m')
type7 = ax.scatter(type7_x,type7_y,type7_z,c='#e24fff')
type8 = ax.scatter(type8_x,type8_y,type8_z,c='#524C90')
type9 = ax.scatter(type9_x,type9_y,type9_z,c='#845868')


ax.set_zlabel('Z') #坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')

ax.legend((type0, type1, type2, type3, type4, type5, type6, type7, type8, type9),
          (u'T-shirt/top', u'Trouser', u'Pullover',
           u'Dress', u'Coat', u'Sandal', u'Shirt',
           u'Sneaker', u'Bag', u'Ankle boot'), loc=0)


plt.show()
