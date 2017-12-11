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
print(list(dict_num.keys()))
print(dict_num.values())
print(dict_num)
idx = np.arange(len(list(dict_num.keys())))
plt.figure(figsize=(13,6))
plt.bar(idx, list(dict_num.values()), 0.5, color='green')
plt.xticks(idx, list(dict_num.keys()))
plt.show()