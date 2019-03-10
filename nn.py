import numpy as np


class NeuralNetwork:
    def __init__(self, in_size: int, out_size: int, hidden_size: int) -> None:
        self.input_size: int = in_size
        self.output_size: int = out_size
        self.hidden_size: int = hidden_size
        self.w1: float = np.random.randn(self.input_size, self.hidden_size)
        self.w2: float = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, x) -> np.array:
        self.z: np.array = np.dot(x, self.w1)
        self.z2: np.array = self.sigmoid(self.z)
        self.z3: np.array = np.dot(self.z2, self.w2)
        o: np.array = self.sigmoid(self.z3)

        return o

    def backward(self, x, y, o) -> None:
        self.o_error: np.array = y - o
        self.o_delta: np.array = self.o_error * self.sigmoid_prime(o)

        self.z2_error: np.array = self.o_delta.dot(self.w2.T)
        self.z2_delta: np.array = self.z2_error * self.sigmoid_prime(self.z2)

        self.w1 += x.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)

    def sigmoid(self, s) -> float:
        return 1 / (1 + np.exp(-s))

    def sigmoid_prime(self, s) -> float:
        return s * (1 - s)

    def train(self, x: np.array, y: np.array) -> None:
        o: np.array = self.forward(x)
        self.backward(x, y, o)
