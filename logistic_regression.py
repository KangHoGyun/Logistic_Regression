import numpy as np
import sys
import matplotlib.pyplot as plt

def sigmoid(z): #sigmoid 함수
    eMin = -np.log(np.finfo(type(0.1)).max)
    zSafe = np.array(np.maximum(z, eMin))
    return(1.0/(1+np.exp(-zSafe)))

class Logistc_Regression:
    def __init__(self, x, y, n):
        x0_list = np.ones((x.shape[0], 1))
        arr_x = np.append(x0_list, x, axis=1) #x데이터에 x0룰 넣어주기 위한 작업입니다.
        self.x = arr_x #(m,n+1)
        self.y = y #(m,1)
        self.w = np.random.rand(self.x.shape[1],) #(n + 1, )
        self.n = n # 타겟 번호

    def predict(self, x_test, y_test):
        x0_list = np.ones((x_test.shape[0], 1))
        arr_x = np.append(x0_list, x_test, axis=1) #x데이터에 x0를 넣어주기 위한 작업입니다.
        hypothesis = sigmoid(np.dot(arr_x, self.w)) #예측 데이터
        cnt = 0
        for i in range(hypothesis.shape[0]):
            if (hypothesis[i] > 0.5 and y_test[i] == True) or (hypothesis[i] < 0.5 and y_test[i] == False):
                cnt = cnt + 1 #예측 데이터 값이 0.5보다 높고 그때의 y값이 참이면 cnt를 +1 해줍니다.
                #또한 예측 데이터 값이 0.5보다 낮고 그때의 y값이 거짓이면 결과를 맞춘것이니까 cnt를 +1 해줍니다.
        print("Accuracy: ", (cnt / hypothesis.shape[0]) * 100, "%")  # 정확도 계산

    def get_cost(self, h):
        h_1 = 1 - h #log 안에 있는 1-h를 h_1이라고 두었습니다.
        for i in range(h.shape[0]):
            if h[i] == 0:
                h[i] += sys.float_info.epsilon #log 안에 있는 h값이 0일 경우에 문제가 발생합니다. 이를 방지하기 위하여 아주 작은 값인 입실론값을 더해줬습니다.
            if h_1[i] == 0:
                h_1[i] += sys.float_info.epsilon #log 안에 있는 1-h값이 0일 경우에 문제가 발생합니다. 이를 방지하기 위하여 아주 작은 값인 입실론값을 더해줬습니다.

        cost = -(np.dot(self.y, np.log(h)) + np.dot((1 - self.y), np.log(h_1)))/h.shape[0] #cost값 계산
        return cost

    def do_learn(self, learning_rate, epoch):
        cost_y = [] # 그래프를 그리기 위해 cost 값을 넣습니다.
        for i in range(epoch): #epoch 만큼 반복
            hypothesis = sigmoid(np.dot(self.x, self.w)) # 예측값
            diff = hypothesis - self.y # 차이
            cost = self.get_cost(hypothesis) #cost값
            for k in range(self.w.shape[0]): #(n)
                xj = self.x[:, k] #(m,1)
                self.w[k] = self.w[k] - learning_rate * np.dot(diff, xj) #grdient descent
            print("epoch : ", i , "  cost: ", cost)
            cost_y.append(cost) # 그래프를 그리기 위해 cost 값을 넣기.
        cost_x = np.arange(0, epoch, 1) # x축
        cost_y = np.array(cost_y) # numpy로 바꾼 후 y축
        plt.plot(cost_x, cost_y, label="target {num}".format(num=self.n)) #plot
        plt.xlabel("number of iterations")
        plt.ylabel("cost")
        plt.legend()
        plt.show()