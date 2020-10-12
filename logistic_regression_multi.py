import numpy as np
import sys
import matplotlib.pyplot as plt

def sigmoid(z): #sigmoid 함수
    eMin = -np.log(np.finfo(type(0.1)).max)
    zSafe = np.array(np.maximum(z, eMin))
    return(1.0/(1+np.exp(-zSafe)))

class Logistc_Regression:
    def __init__(self, x, y, n):
        num = np.unique(y, axis=0)
        num = num.shape[0]
        y = np.eye(num)[y] # one-hot encoding
        x0_list = np.ones((x.shape[0], 1))
        arr_x = np.append(x0_list, x, axis=1) #x데이터에 x0룰 넣어주기 위한 작업입니다.
        self.x = arr_x #(m,n +1) m은 데이터 수, n은 피쳐 수, t는 타겟 수
        self.y = y #one-hot encoding (m, t)
        self.w = np.random.rand(self.x.shape[1], n) #(n + 1, t)

    def predict(self, x_test, y_test):
        num = np.unique(y_test, axis=0)
        num = num.shape[0]
        y_test = np.eye(num)[y_test] #one-hot encoding
        x0_list = np.ones((x_test.shape[0], 1))
        arr_x = np.append(x0_list, x_test, axis=1) #x 데이터에 x0로 1을 넣어주기 위한 작업입니다.
        hypothesis = sigmoid(np.dot(arr_x, self.w)) # 예측값
        cnt = 0
        for i in range(hypothesis.shape[0]):
            index = np.argmax(hypothesis[i]) #가장 높은 예측값의 인덱스를 뽑아냅니다.
            if y_test[i][index] == True: #이때 그 인덱스의 y값이 참일 경우에 cnt +1 을 해줍니다.
                cnt = cnt + 1
        print("Accuracy: ", (cnt / hypothesis.shape[0]) * 100, "%")  # 정확도 계산

    def get_cost(self, h):
        h_1 = 1 - h #log 안에 있는 1-h를 h_1이라고 두었습니다.
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                if h[i][j] == 0:
                    h[i][j] += sys.float_info.epsilon #log 안에 있는 h값이 0일 경우에 문제가 발생합니다. 이를 방지하기 위하여 아주 작은 값인 입실론값을 더해줬습니다.
                if h_1[i][j] == 0:
                    h_1[i][j] += sys.float_info.epsilon #log 안에 있는 1-h값이 0일 경우에 문제가 발생합니다. 이를 방지하기 위하여 아주 작은 값인 입실론값을 더해줬습니다.
        cost = -np.sum(self.y * np.log(h) + (1 - self.y) * np.log(h_1), axis=0) / self.y.shape[0] #cost 값 계산
        return cost

    def do_learn(self, learning_rate, epoch):
        cost_y = []  # 그래프를 그리기 위해 cost 값을 넣습니다.
        for i in range(epoch): #epoch 만큼 반복
            hypothesis = sigmoid(np.dot(self.x, self.w)) # 예측값, (m, t)
            diff = hypothesis - self.y # 실제 값과의 차이(m,t)
            cost = self.get_cost(hypothesis) # cost값
            diff2 = np.transpose(diff) # gradient descent 를 하기 위해 차이 값을 (m, t)에서 (t, m)으로 바꿔주었습니다.
            for k in range(self.w.shape[0]): #(n)
                xj = self.x[:, k] #xj를 reshape (m, )
                self.w[k] = self.w[k] - learning_rate * np.dot(diff2, xj) #grdient descent
            print("epoch : ", i , "  cost: ", cost)
            cost_y.append(cost)  # 그래프를 그리기 위해 cost 값을 넣기.
        cost_x = np.arange(0, epoch, 1)  # x축
        cost_y = np.array(cost_y)  # numpy로 바꾼 후 y축
        plt.plot(cost_x, cost_y)  # plot
        plt.xlabel("number of iterations")
        plt.ylabel("cost")
        plt.show()