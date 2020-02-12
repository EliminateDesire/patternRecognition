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
X = np.vstack((boyHeight,boyWeight))
boy_mu = np.array([boyHeight.mean(),boyWeight.mean()])

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
girl_mu = np.array([girlHeight.mean(),girlWeight.mean()])

sBoy = 0
sGirl = 9
for i in range(len(heiB)):
    sBoy += np.matmul((np.array([heiB[i],weiB[i]]) - boy_mu).reshape(2,1),(np.array([heiB[i],weiB[i]]) - boy_mu).reshape(1,2))
for i in range(len(heiG)):
    sGirl += np.matmul((np.array([heiG[i],weiG[i]]) - girl_mu).reshape(2,1),(np.array([heiG[i],weiG[i]]) - girl_mu).reshape(1,2))
Sw = sBoy + sGirl
W = np.matmul(np.linalg.inv(Sw), (boy_mu - girl_mu).reshape(2,1))
list3 = []
y = []
with open('boy2017.txt', mode='r', encoding='utf-8') as fi:
    lines = fi.readlines()
for line in lines:
    line = line.split()
    list3.append([float(line[0]), float(line[1])])
    y.append(1)
with open('girl2017.txt', mode='r', encoding='utf-8') as fi:
    lines = fi.readlines()
lines.pop(-1)
for line in lines:
    line = line.split()
    list3.append([float(line[0]), float(line[1])])
    y.append(0)
X = np.array(list3)


def classify(data,W):
    data = np.array(data).reshape(2,1)
    result = np.matmul(W.T,data - 0.5 * (boy_mu.reshape(2,1) + girl_mu.reshape(2,1)))
    if(result > 0):
        return 1
    else:
        return 0

def plot_decision_boundary2D(X, y, num_row=200, num_col=200):
    """
    绘制决策边界的核心代码
    :param clf: 分类器， 即使用的模型
    :param X: 输入的数据X
    :param y: 真实的分类结果y
    :param num_row: 绘制决策边界时，行数据生成的个数
    :param num_col: 列数据生成的个数
    """
    sigma = 1  # 防止数据在图形的边上而加上的一个偏移量，设定一个较小的值即可
    x1_min, x1_max = np.min(X[:, 0]) - sigma, np.max(X[:, 0])
    x2_min, x2_max = np.min(X[:, 1]) - sigma, np.max(X[:, 1])
    # print(x1_min)
    t1 = np.linspace(x1_min, x1_max, num_row)
    t2 = np.linspace(x2_min, x2_max, num_col)
    x1, x2 = np.meshgrid(t1, t2)
    # print(x1)
    x_test = np.stack((x1.flat, x2.flat), axis=1)

    # 设置使用的颜色colors， 这里假设最后的结果是三个类别
    cm_dark = mpl.colors.ListedColormap(['#FFA0A0', '#A0A0FF'])
    cm_light = mpl.colors.ListedColormap(['r', 'b'])

    y_hat = []
    # print(x_test.shape)
    # print(x_test)
    for data in x_test:
        result = classify(data, W)
        if result == 1:
            y_hat.append(np.int8(0))
        else:
            y_hat.append(np.int8(1))
    y_hat = np.array(y_hat)
    #print(x1.shape)
    # y_hat = y_hat.reshape(x1.shape[0])
    #print(y_hat.shape, ' ', y_hat)
    plt.pcolormesh(x1, x2, y_hat.reshape(x1.shape), cmap=cm_dark)  # 绘制底色
    plt.scatter(X[:, 0], X[:, 1], s=20, c=y, edgecolors='k', cmap=cm_light)  # 绘制数据的颜色
    #print(X[:, 0])

    plt.xlabel("身高")
    plt.ylabel("体重")
    plt.title('decision-boundary')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.plot([0,W[0]],[0,W[1]])
    plt.show()

if __name__ == '__main__':
   # print(X[0])
   # print(W)
    plot_decision_boundary2D(X, np.array(y))
    plt.plot([0, W[0]], [0, W[1]])
   # print(W[0],W[1])
   # plt.plot([0,0],[W[0] * 10000,W[1] * 100000])
   # plt.show()