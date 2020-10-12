import numpy as np
from tlnn2 import TwoLayerNeuralNetwork2

input_size = 2
hidden_size =3 #hidden size를 바꿔가면서 해보자
output_size = 1
tn2 = TwoLayerNeuralNetwork2(input_size, hidden_size, output_size)

print(tn2.params)
x = np.array([0.5, 0.8])
print(tn2.predict(x))