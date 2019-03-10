import numpy as np
from nn import NeuralNetwork

x: np.array = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y: np.array = np.array(([92], [86], [89]), dtype=float)

x: np.array = x / np.amax(x, axis=0)
y: np.array = y / 100

neural_net: NeuralNetwork = NeuralNetwork(x.shape[1], 1, x.shape[0])
o: np.array = neural_net.forward(x)

print(f'Input:\n {x}')
print(f'Actual:\n {y}')
for i in range(0, 1000):
    predicted: np.array = neural_net.forward(x)
    loss: float = np.mean(np.square(y - neural_net.forward(x)))
    neural_net.train(x, y)
print(f'Predicted:\n {predicted}')
print(f'Final Loss: {loss}')
print(f'Final error:\n{y - predicted}')