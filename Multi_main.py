import numpy as np
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from dataset.mnist import load_mnist
from logistic_regression_multi import Logistc_Regression

iris = load_iris()

X = iris.data # iris data input
y = iris.target # iris target (label)
y_name = iris.target_names # iris target name

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False) #mnist

label_name = ['0','1','2','3','4','5','6','7','8','9'] #mnist label

# 150개 랜덤하게 뒤섞은 후에 학습데이터와 테스트데이터의 비율을 7:3으로 나눔.
num = int(X.shape[0]/10*7)
select = np.random.permutation(150) # 0~149 숫자들을 랜덤하게 뒤섞음
Xtr = X[select[:num]] # 105개
Xte = X[select[num:]] # 45개
Ytr = y[select[:num]]
Yte = y[select[num:]]

Iris = Logistc_Regression(Xtr, Ytr, len(y_name))
print("Iris Output: Multiple Class")
Iris.do_learn(0.001, 1000)
Iris.predict(Xte, Yte)

Mnist = Logistc_Regression(x_train, t_train, len(label_name))
print("MNIST Output: Multiple Class")
Mnist.do_learn(0.001, 100)
Mnist.predict(x_test, t_test)