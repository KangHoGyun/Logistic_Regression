import numpy as np
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from dataset.mnist import load_mnist
from logistic_regression import Logistc_Regression

iris = load_iris()

X = iris.data # iris data input
y = iris.target # iris target (label)
y_name = iris.target_names # iris target name

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False) #mnist

label_name = ['0','1','2','3','4','5','6','7','8','9'] #mnist label
tr_list = []
for i in range(len(label_name)):
    temp_list = list()
    for j in t_train:
        if i == j:
            temp_list.append(True)
        else:
            temp_list.append(False)
    tr_list.append(temp_list)
tr_list = np.array(tr_list)

te_list = []
for i in range(len(label_name)):
    temp_list = list()
    for j in t_test:
        if i == j:
            temp_list.append(True)
        else:
            temp_list.append(False)
    te_list.append(temp_list)
te_list = np.array(te_list)




y_list = []
for i in range(y_name.shape[0]):
    temp_list = list()
    for j in y:
        if i == j:
            temp_list.append(True)
        else:
            temp_list.append(False)
    y_list.append(temp_list)
y_list = np.array(y_list)
# 150개 랜덤하게 뒤섞은 후에 학습데이터와 테스트데이터의 비율을 7:3으로 나눔.
num = int(X.shape[0]/10*7)
select = np.random.permutation(150) # 0~149 숫자들을 랜덤하게 뒤섞음
Xtr = X[select[:num]] # 105개
Xte = X[select[num:]] # 45개

ytr_0 = y_list[0][select[:num]]
ytr_1 = y_list[1][select[:num]]
ytr_2 = y_list[2][select[:num]]

yte_0 = y_list[0][select[num:]]
yte_1 = y_list[1][select[num:]]
yte_2 = y_list[2][select[num:]]


x0_list = np.ones((Xte.shape[0], 1))
arr_x = np.append(x0_list, Xte, axis=1)

print(arr_x)