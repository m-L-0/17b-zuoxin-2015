#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : pie.py
# @Author: Joker
# @Date  : 2017/12/4

import matplotlib.pyplot as plt
import numpy as np
import os


plt.rcParams['font.sans-serif'] = ['SimHei']
root = os.pardir+os.sep+"car"+os.sep
dict_num = {}
for rt, dirs, files in os.walk(root):
    for f in files:
        p, lab = os.path.split(rt)
        if dict_num.get(lab) != None:
            dict_num[lab]+=1
        else:
            dict_num[lab]=0

labels = dict_num.keys()
fracs = [a/sum(dict_num.values()) for a in list(dict_num.values())]
explode = np.zeros(len(dict_num.keys()))
plt.axes(aspect=1)
plt.pie(x=fracs, labels=labels, explode=explode, autopct='%3.1f %%',
        shadow=True, labeldistance=1.1, startangle=90, pctdistance=0.6)
plt.show()