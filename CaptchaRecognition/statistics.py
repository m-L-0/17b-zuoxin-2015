#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : statistics.py
# @Author: Joker
# @Date  : 2017/12/12

import csv

csvfile = open('captcha/labels/labels.csv', 'r')
reader = csv.reader(csvfile)
dict_len = {1: 0, 2: 0, 3: 0, 4: 0}
for i, j in reader:
    dict_len[len(str(j))] += 1

print(dict_len)
