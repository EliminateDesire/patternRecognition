import numpy as np
import matplotlib.pyplot as plt
import math
import codecs

with codecs.open('boy.txt', mode='r', encoding='utf-8') as fi:
    lines = fi.readlines()

list1, list2, list3 = [], [], []
heiB, weiB = [], []
heiG, weiG = [], []
for line in lines:
    line = line.split()
    list1.append(line[0])
    list2.append(line[1])
    list3.append(line[2])
    heiB.append(float(line[0]))
    weiB.append(float(line[1]))
boyHeight = np.array(list1, dtype='float32')
boyWeight = np.array(list2, dtype='float32')
boyShoeSize = np.array(list3, dtype='float32')
X = np.vstack((boyHeight, boyWeight, boyShoeSize))
X2 = np.vstack((boyHeight, boyWeight))
boy_theta = np.cov(X)
boy_theta2 = np.cov(X2)
boy_mu = np.array([boyHeight.mean(), boyWeight.mean(), boyShoeSize.mean()])
boy_mu2 = np.array([boyHeight.mean(), boyWeight.mean()])

with codecs.open('girl.txt', mode='r', encoding='utf-8') as fi:
    lines = fi.readlines()
list1.clear()
list2.clear()
list3.clear()
for line in lines:
    line = line.split()
    list1.append(line[0])
    list2.append(line[1])
    list3.append(line[2])
    heiG.append(float(line[0]))
    weiG.append(float(line[1]))
girlHeight = np.array(list1, dtype='float32')
girlWeight = np.array(list2, dtype='float32')
girlShoeSize = np.array(list3, dtype='float32')
XX = np.vstack((girlHeight, girlWeight, girlShoeSize))
XX2 = np.vstack((girlHeight, girlWeight))
girl_theta = np.cov(XX)
girl_theta2 = np.cov(XX2)
girl_mu = np.array([girlHeight.mean(), girlWeight.mean(), girlShoeSize.mean()])
girl_mu2 = np.array([girlHeight.mean(), girlWeight.mean()])
err = 0


def test(boy_mu, boy_theta, girl_mu, girl_theta):
    global err
    with open('boy2017.txt', mode='r', encoding='utf-8') as fi:
        lines = fi.readlines()
    for line in lines:
        h, w, s = map(float, line.split())
        x = np.array([h, w, s])
        gb = -0.5 * np.matmul(np.matmul((x - boy_mu), np.linalg.inv(boy_theta)),
                              (x - boy_mu).reshape(3, 1)) - 0.5 * math.log(np.linalg.det(boy_theta)) + math.log(0.5)
        gg = -0.5 * np.matmul(np.matmul((x - girl_mu), np.linalg.inv(girl_theta)),
                              (x - girl_mu).reshape(3, 1)) - 0.5 * math.log(np.linalg.det(girl_theta)) + math.log(0.5)
        if (gg > gb):
            err += 1

    with open('girl2017.txt', mode='r', encoding='utf-8') as fi:
        lines = fi.readlines()
    lines.pop(-1)
    for line in lines:
        h, w, s = map(float, line.split())
        x = np.array([h, w, s])
        gb = -0.5 * np.matmul(np.matmul((x - boy_mu), np.linalg.inv(boy_theta)),
                              (x - boy_mu).reshape(3, 1)) - 0.5 * math.log(np.linalg.det(boy_theta))
        gg = -0.5 * np.matmul(np.matmul((x - girl_mu), np.linalg.inv(girl_theta)),
                              (x - girl_mu).reshape(3, 1)) - 0.5 * math.log(np.linalg.det(girl_theta))
        if (gb > gg):
            err += 1


def show(boy_mu2, boy_theta2, girl_mu2, girl_theta2):
    H, W = [], []
    for h in range(0, 400):
        for w in range(0, 400):
            x = np.array([h, w])
            gb = -0.5 * np.matmul(np.matmul((x - boy_mu2), np.linalg.inv(boy_theta2)),
                                  (x - boy_mu2).reshape(2, 1)) - 0.5 * math.log(np.linalg.det(boy_theta2))
            gg = -0.5 * np.matmul(np.matmul((x - girl_mu2), np.linalg.inv(girl_theta2)),
                                  (x - girl_mu2).reshape(2, 1)) - 0.5 * math.log(np.linalg.det(girl_theta2))
            if (gb - gg < 0.00001):
                H.append(h)
                W.append(w)
    plt.scatter(H, W, c='b')
    plt.scatter(heiB, weiB, c='r')
    plt.scatter(heiG, weiG, c='y')
    plt.show()


def scanf(boy_mu, boy_theta, girl_mu, girl_theta):
    while True:
        print("请输入身高，体重，鞋码")
        h, w, s = map(float, input().split())
        x = np.array([h, w, s])
        gb = -0.5 * np.matmul(np.matmul((x - boy_mu), np.linalg.inv(boy_theta)),
                              (x - boy_mu).reshape(3, 1)) - 0.5 * math.log(np.linalg.det(boy_theta))
        gg = -0.5 * np.matmul(np.matmul((x - girl_mu), np.linalg.inv(girl_theta)),
                              (x - girl_mu).reshape(3, 1)) - 0.5 * math.log(np.linalg.det(girl_theta))
        if (gb > gg):
            print("男的")
        else:
            print("女的")


if __name__ == '__main__':
    # test(boy_mu,boy_theta,girl_mu,girl_theta)
    # scanf(boy_mu,boy_theta,girl_mu,girl_theta)
    show(boy_mu2, boy_theta2, girl_mu2, girl_theta2)
