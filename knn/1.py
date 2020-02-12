import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import codecs
with codecs.open('boy.txt',mode = 'r',encoding = 'utf-8') as fi:
    lines = fi.readlines()

list1,list2 = [],[]
heiB,weiB = [],[]
heiG,weiG = [],[]
for line in lines:
    line = line.split()
    list1.append(line[0])
    list2.append(line[1])
    heiB.append(float(line[0]))
    weiB.append(float(line[1]))
boyHeight = np.array(list1,dtype = 'float32')
boyWeight = np.array(list2,dtype = 'float32')

with codecs.open('girl.txt',mode = 'r',encoding = 'utf-8') as fi:
    lines = fi.readlines()
list1.clear()
list2.clear()
for line in lines:
    line = line.split()
    list1.append(line[0])
    list2.append(line[1])
    heiG.append(float(line[0]))
    weiG.append(float(line[1]))
girlHeight = np.array(list1,dtype = 'float32')
girlWeight = np.array(list2,dtype = 'float32')

def classify(k,hei,wei):
    dis = []
    lab = []
    for i in range(len(boyHeight)):
        dis.append(math.sqrt((boyHeight[i] - hei) ** 2 + (boyWeight[i] - wei) ** 2))
        lab.append(1)
    for i in range(len(girlHeight)):
        dis.append(math.sqrt((girlHeight[i] - hei) ** 2 + (girlWeight[i] - wei) ** 2))
        lab.append(0)
    dis = np.array(dis)
    lab = np.array(lab)
    lab = lab[np.argsort(dis)]
    sum_boy = 0
    sum_girl = 0
    for i in range(0,k):
        if lab[i] == 1:
            sum_boy += 1
        else:
            sum_girl += 1
    if(math.fabs(sum_girl - sum_boy) <= 0):
        return 0
    else:
        return 1

def show():
    H, W = [], []
    for h in range(50, 185):
        for w in range(40, 90):
            if (classify(4,h,w) == 0):
                H.append(h)
                W.append(w)
    plt.scatter(H, W, c='b')
    plt.scatter(heiB, weiB, c='r')
    plt.scatter(heiG, weiG, c='y')
    plt.show()

if __name__ == '__main__':
    show()
